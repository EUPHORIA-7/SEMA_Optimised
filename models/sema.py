import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
import math
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SEMAVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbone.sema_block import SEMAModules
from torch.cuda.amp import GradScaler, autocast
import copy

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        
        self.scaler = GradScaler()
        self._old_network = None

    def after_task(self):
        self._known_classes = self._total_classes
        
        # [LwF] Save old model for distillation
        logging.info("Caching old model for distillation...")
        self._old_network = self._network.copy().freeze()
        self._old_network.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Pass task ID to blocks
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.set_cur_task(self._cur_task)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def replace_fc(self, train_loader):
        """
        NCM (Prototype) Logic
        """
        logging.info("Computing class prototypes (NCM) for new classes...")
        self._network.eval()
        embedding_list = []
        label_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                
                # Get features directly
                outcome = self._network(data)
                embedding = outcome['features']
                
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(label_list.numpy())
        
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=False).squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            
            self._network.fc.weight.data[class_index] = proto.to(self._device)
            bias=-0.5*torch.sum(proto**2)
            if self._network.fc.bias is not None:
                self._network.fc.bias.data[class_index]=bias.to(self._device) 
            
        logging.info("Prototype replacement complete.")

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0:
            self._train_new(train_loader, test_loader)
            self.replace_fc(train_loader) # NCM for Task 0
        else:
            # Standard SEMA Logic (No unfreezing)
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True
            
            detect_loader = DataLoader(train_loader.dataset, batch_size=self.args["detect_batch_size"], shuffle=True, num_workers=num_workers)     
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False
            
            if added == 0:
                self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
                self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
            
            # Apply NCM after training
            self.replace_fc(train_loader)
            
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
        self._init_train(self.args['func_epoch'], train_loader, test_loader, self.optimizer, self.scheduler, phase='func')
        self.update_rd_optimizer_and_scheduler(num_epoch=self.args['rd_epoch'], lr=self.args['rd_lr'])
        self._init_train(self.args['rd_epoch'], train_loader, test_loader, self.rd_optimizer, self.rd_scheduler, phase='rd')

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        is_added = False
        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            with torch.no_grad():
                model_outcome = self._network(inputs)
            
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:
                added += 1
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False
                self._train_new(train_loader, test_loader)
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()
        
        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        else:
            return added

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):
        prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct = torch.tensor(0, dtype=torch.int64)
            total = 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                optimizer.zero_grad()
                
                with autocast():
                    if phase == "func":
                        outcome = self._network(inputs)
                        logits = outcome["logits"]
                        logits = logits[:, :self._total_classes]
                        
                        # 2. [LwF] Distillation Loss
                        loss_kd = torch.tensor(0.).to(self._device)
                        if self._cur_task > 0 and self._old_network is not None:
                            with torch.no_grad():
                                old_outcome = self._old_network(inputs)
                                old_logits = old_outcome["logits"]
                            
                            # Distill on old classes
                            loss_kd = _KD_loss(
                                logits[:, :self._known_classes],
                                old_logits[:,:self._known_classes],
                                T=2.0
                            )
                        
                        logits_for_ce=logits.clone()
                        if self._cur_task > 0:
                           logits[:, :self._known_classes] = -float('inf')

                        # 1. CE Loss
                        loss_clf = F.cross_entropy(logits, targets)
                        
                        loss = loss_clf + loss_kd

                        _, preds = torch.max(logits, dim=1)
                        correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                        total += len(targets)

                    elif phase == "rd":
                        outcome = self._network(inputs)
                        logits = outcome["logits"] 
                        loss = outcome["rd_loss"]

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                losses += loss.item()

            if scheduler is not None:
                scheduler.step()
            
            if phase == "func" and total > 0:
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            else:
                train_acc = 0.0

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                phase, self._cur_task, epoch + 1, total_epoch,
                losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [p for n,p in self._network.named_parameters() if ('functional' in n or 'router' in n or 'fc' in n or 'q_proj' in n or 'k_proj' in n or 'temperature' in n) and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.optimizer = optim.SGD(func_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.optimizer = optim.AdamW(func_params, lr=lr, weight_decay=self.args["weight_decay"])

        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=min_lr)    

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["rd_lr"] if lr is None else lr
        rd_params = [p for n,p in self._network.named_parameters() if 'rd' in n and p.requires_grad]
        if self.args['optimizer']=='sgd':
            self.rd_optimizer = optim.SGD(rd_params, momentum=0.9, lr=lr,weight_decay=self.args["weight_decay"])
        elif self.args['optimizer']=='adam':
            self.rd_optimizer = optim.AdamW(rd_params, lr=lr, weight_decay=self.args["weight_decay"])
        
        min_lr = self.args['min_lr'] if self.args['min_lr'] is not None else 1e-8
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch, eta_min=min_lr) if self.rd_optimizer else None
        
    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if 'adapter' in k or ('fc' in k and 'block' not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]