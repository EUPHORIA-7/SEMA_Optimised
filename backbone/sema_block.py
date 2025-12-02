import torch
from torch import nn
from typing import List
import copy
import logging
from backbone.sema_components import Adapter, AE, Records

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class AdapterModule(nn.Module):    
    def __init__(self, config, adapter_id, writer):
        super().__init__()
        self.config = config
        self.functional = Adapter(self.config, adapter_id, dropout=0.1, bottleneck=self.config.ffn_num,
                                init_option=self.config.ffn_adapter_init_option,
                                adapter_scalar=self.config.ffn_adapter_scalar,
                                adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
                                )
        layer_id = int(adapter_id.split('.')[0])
        self.not_addition_layer = layer_id < config.adapt_start_layer or layer_id > config.adapt_end_layer
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
        self.activation = nn.ReLU()
        self.newly_added = True
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def forward(self, x):
        func_out = self.functional(x)
        if self.not_addition_layer:
            rd_loss = torch.tensor(0.).to(device)
            return func_out, rd_loss, torch.zeros_like(rd_loss).to(device)
        else:
            rd_loss = self.rd.compute_reconstruction_loss(x)
        z_score = self.get_z_score_deviation(rd_loss)
        if self.training:
            self.add_z_score_record(rd_loss)
        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev
        if not self.rd_loss_record.length > 2:
            return torch.zeros_like(rd_loss).to(device)
        z_score = (rd_loss-mean)/stddev
        z_score = torch.abs(z_score)
        return z_score
    
    def add_z_score_record(self, rd_loss):
        self.rd_loss_record.add_record(rd_loss.detach().cpu())


class SEMAModules(nn.Module):
    def __init__(self, config, layer_id, writer):
        super().__init__()
        self.adapters: List[Adapter] = nn.ModuleList()
        self.config = config
        self.act_func = nn.ReLU()
        self.layer_id = layer_id
        self.writer = writer
        self.newly_added = True
        self.added_for_task = True
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        
        # Initialize with one adapter
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        # --- [FEATURE 1: Attention-Based Router] ---
        # Default to 128 if not in config
        self.attention_dim = getattr(self.config, "router_attn_dim", 128)
        
        # Query projection (from input features)
        self.q_proj = nn.Linear(config.d_model, self.attention_dim).cuda()
        
        # Key projection (from adapter's static weights)
        self.k_proj = nn.Linear(config.d_model, self.attention_dim).cuda()
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Placeholder for legacy support
        self.router = nn.Linear(config.d_model, 1).cuda() 
        self.new_router = None
        # --- [END FEATURE 1] ---

        self.detecting_outlier = False
        
        # --- [FEATURE 2: Adaptive Thresholding] ---
        self.z_score_record = Records(max_len=config.buffer_size) 
        # Default to 3.0 if not in config
        self.exp_k_std = getattr(self.config, "exp_k_std", 3.0)
        self._cur_task = 0
        # --- [END FEATURE 2] ---
        
    @property
    def num_adapters(self):
        return len(self.adapters)

    def set_cur_task(self, task_id):
        self._cur_task = task_id
       
    def fix_router(self):
        # Legacy method kept to prevent crashes if called
        pass

    def add_adapter(self, initialize=False):
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"
        new_adapter = AdapterModule(self.config, adapter_id, self.writer).to(device)
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        logging.info(f"Adapter {adapter_id} added at block {self.layer_id}")

    def forward(self, x):
        rd_loss = torch.tensor(0.).to(device)
        added = False
        not_addition_layer = self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer
        
        if not_addition_layer:
            func_out, _, _= self.adapters[-1](x)
        else:
            func_outs, rd_losses, z_scores = [], [], []
            for adapter in self.adapters:
                func_out, rd_loss, z_score = adapter(x)
                func_outs.append(func_out)  
                rd_losses.append(rd_loss)   
                z_scores.append(z_score)

            func_outs = torch.stack(func_outs)
            rd_losses = torch.stack(rd_losses)
            z_scores = torch.stack(z_scores)
            
            # --- [LOGIC: Adaptive Thresholding] ---
            min_z_score = z_scores.mean(dim=1).min()
            
            if self.z_score_record.length > 10: 
                dynamic_threshold = self.z_score_record.mean + self.exp_k_std * self.z_score_record.stddev
                is_dynamic = True
            else:
                dynamic_threshold = self.config.exp_threshold 
                is_dynamic = False

            # Debug print for monitoring (only print for first relevant layer to avoid spam)
            if self.layer_id == self.config.adapt_start_layer and self.detecting_outlier:
                print(f"[Task {self._cur_task}, Layer {self.layer_id}] "
                      f"Min Z-Score: {min_z_score.item():.4f} | "
                      f"Threshold: {dynamic_threshold:.4f} (Dynamic: {is_dynamic}) | "
                      f"Buffer Mean/Std: {self.z_score_record.mean:.4f} / {self.z_score_record.stddev:.4f}")

            addition_criteria = min_z_score > dynamic_threshold \
                and self.layer_id >= self.adapt_start_layer \
                and self.layer_id <= self.adapt_end_layer \
                and not self.added_for_task and self.detecting_outlier

            if addition_criteria:
                self.add_adapter()
                out = {"func_out": torch.zeros_like(func_outs[0]).to(device), "rd_loss": torch.tensor(0.).to(device), "added": True}
                return out
            else:
                # Record z-score only if we are detecting and NOT adding
                if self.detecting_outlier and not self.added_for_task:
                    self.z_score_record.add_record(min_z_score.detach().cpu().unsqueeze(0))

                # --- [LOGIC: Attention Routing] ---
                # 1. Query (Q): [batch, attention_dim]
                q_feat = x.mean(dim=1)
                q = self.q_proj(q_feat)

                # 2. Keys (K): [num_adapters, attention_dim]
                keys = []
                for adapter in self.adapters:
                    # Use mean of down_proj weights as key source
                    # Detach to ensure we don't backprop into adapter weights via the key
                    adapter_key_src = adapter.functional.down_proj.weight.mean(dim=0).detach()
                    keys.append(self.k_proj(adapter_key_src)) 
                
                k = torch.stack(keys, dim=0)

                # 3. Attention Scores: [batch, num_adapters]
                attn_scores = torch.matmul(q.unsqueeze(1), k.transpose(0, 1)) / (self.temperature + 1e-6)
                attn_scores = attn_scores.squeeze(1)
                
                mask = torch.softmax(attn_scores, dim=1)
                # --- [END Attention Routing] ---
                
                func_out = (func_outs * mask.transpose(0,1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
                
                if self.adapters[-1].newly_added:
                    rd_loss = rd_losses[-1].mean()
                else:
                    rd_loss = torch.tensor(0.).to(device)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False
        
        # Reset z-score stats for the next task
        self.z_score_record = Records(max_len=self.config.buffer_size)

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        # Only freeze the adapters themselves.
        # DO NOT freeze the router (q_proj, k_proj, temperature)
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None

    def freeze_rd(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False