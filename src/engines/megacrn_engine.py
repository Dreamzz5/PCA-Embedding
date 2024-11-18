import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
from torch import nn

class MegaCRN_Engine(BaseEngine):
    def __init__(self, step_size, horizon, lamb=0.01, lamb1=0.01, lamb2=0.01, **args):
        super(MegaCRN_Engine, self).__init__(**args)
        self._step_size = step_size
        self._horizon = horizon
        self._task_level = 0
        # Loss weighting parameters
        self.lamb = lamb    # Weight for separate loss
        self.lamb1 = lamb1  # Weight for compact loss
        self.lamb2 = lamb2  # Weight for attention consistency loss
        
    def megacrn_loss(self, query, pos, neg, null_val):
        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()
        loss2 = separate_loss(query, pos.detach(), neg.detach())
        loss3 = compact_loss(query, pos.detach())
        loss =  0.01 * loss2 + 0.01 * loss3

        return loss

    def train_batch(self):
        self.model.train()
        
        train_loss = []
        train_mape = []
        train_rmse = []
        
        self._dataloader['train_loader'].shuffle()
        
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            
            if self._iter_cnt % self._step_size == 0 and self._task_level < self._horizon:
                self._task_level += 1
            
            # Prepare data
            node_embed = self._dataloader["train_loader"].node_embed
            X, label, node_embed = self._to_device(
                self._to_tensor([X, label, node_embed])
            )
            
            # Forward pass
            node_embed = torch.nn.functional.dropout(node_embed, 0.15, training=True)
            pred, h_att, query, pos, neg = self.model(X, label, node_embed, self._iter_cnt, self._task_level)
            pred, label = self._inverse_transform([pred, label])
            
            # Handle precision issues
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            
            # Calculate total loss with all components
            loss = self._loss_fn(pred, label, mask_value)
            # Calculate metrics
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            
            # Backward pass
            (loss + self.megacrn_loss(query, pos, neg, mask_value)).backward()
            
            # Gradient clipping if needed
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            
            self._optimizer.step()
            
            # Store metrics
            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            
            self._iter_cnt += 1
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)