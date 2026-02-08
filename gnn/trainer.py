"""
Training utilities for the GNN Survival Pipeline
"""

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Batch
from typing import List, Dict, Tuple
import time


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
    
    def load_best_model(self, model: torch.nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class GNNTrainer:
    """
    Trainer for GNN survival models.
    
    Handles:
    - Training loop with batching
    - Validation and early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Concordance index evaluation
    """
    
    def __init__(self, model: torch.nn.Module, config, criterion):
        self.model = model
        self.config = config
        self.criterion = criterion
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize C-index
            patience=config.PATIENCE // 2,
            factor=config.LR_FACTOR,
            min_lr=config.MIN_LR,
            verbose=False
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode='max'
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_c_index': [],
            'lr': []
        }
    
    def train_epoch(self, train_graphs: List, verbose: bool = False) -> float:
        """
        Train for one epoch.
        
        Args:
            train_graphs: List of PyTorch Geometric Data objects
            verbose: Print batch progress
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        # Shuffle
        indices = np.random.permutation(len(train_graphs))
        train_graphs = [train_graphs[i] for i in indices]
        
        # Create batches
        batch_size = self.config.BATCH_SIZE
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i + batch_size]
            batch = Batch.from_data_list(batch_graphs).to(self.config.DEVICE)
            
            # Forward
            self.optimizer.zero_grad()
            risk = self.model(batch)
            
            # Loss
            loss = self.criterion(risk, batch.survival_time, batch.event)
            
            if torch.isnan(loss):
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if verbose and n_batches % 10 == 0:
                print(f"    Batch {n_batches}: Loss = {loss.item():.4f}")
        
        return total_loss / max(n_batches, 1)
    
    def evaluate(self, graphs: List) -> Tuple[float, float]:
        """
        Evaluate model on a set of graphs.
        
        Returns:
            c_index: Concordance index
            loss: Average loss
        """
        self.model.eval()
        
        all_risks = []
        all_times = []
        all_events = []
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            batch_size = self.config.BATCH_SIZE * 2  # Larger batches for eval
            for i in range(0, len(graphs), batch_size):
                batch_graphs = graphs[i:i + batch_size]
                batch = Batch.from_data_list(batch_graphs).to(self.config.DEVICE)
                
                risk = self.model(batch)
                
                # Compute loss
                loss = self.criterion(risk, batch.survival_time, batch.event)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    n_batches += 1
                
                # Collect predictions
                all_risks.extend(risk.cpu().numpy().flatten())
                all_times.extend(batch.survival_time.cpu().numpy())
                all_events.extend(batch.event.cpu().numpy())
        
        # Compute C-index
        c_index = self._concordance_index(
            np.array(all_risks),
            np.array(all_times),
            np.array(all_events)
        )
        
        avg_loss = total_loss / max(n_batches, 1)
        
        return c_index, avg_loss
    
    def train(self, train_graphs: List, val_graphs: List, 
              verbose: bool = True) -> Dict:
        """
        Full training loop with validation.
        
        Args:
            train_graphs: Training graphs
            val_graphs: Validation graphs
            verbose: Print progress
            
        Returns:
            Training history
        """
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            # Train
            train_loss = self.train_epoch(train_graphs)
            
            # Validate
            val_c_index, val_loss = self.evaluate(val_graphs)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_c_index'].append(val_c_index)
            self.history['lr'].append(current_lr)
            
            # Scheduler step
            self.scheduler.step(val_c_index)
            
            # Early stopping
            self.early_stopping(val_c_index, self.model)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, "
                      f"Val C-index={val_c_index:.4f}, LR={current_lr:.6f}")
            
            if self.early_stopping.early_stop:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.early_stopping.load_best_model(self.model)
        
        # Final evaluation
        final_c_index, _ = self.evaluate(val_graphs)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"    Training complete in {elapsed:.1f}s")
            print(f"    Best C-index: {self.early_stopping.best_score:.4f}")
        
        return self.history
    
    def _concordance_index(self, risk: np.ndarray, time: np.ndarray, 
                          event: np.ndarray) -> float:
        """
        Compute Harrell's concordance index.
        
        The C-index measures the proportion of concordant pairs among
        all permissible pairs (where we can determine ordering).
        
        Args:
            risk: Predicted risk scores
            time: Observed survival times
            event: Event indicators (1=event, 0=censored)
            
        Returns:
            C-index between 0 and 1 (0.5 = random)
        """
        n = len(risk)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if both censored
                if event[i] == 0 and event[j] == 0:
                    continue
                
                # Case 1: i died before j and i had event
                if time[i] < time[j] and event[i] == 1:
                    permissible += 1
                    if risk[i] > risk[j]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
                
                # Case 2: j died before i and j had event
                elif time[j] < time[i] and event[j] == 1:
                    permissible += 1
                    if risk[j] > risk[i]:
                        concordant += 1
                    elif risk[i] == risk[j]:
                        concordant += 0.5
                
                # Case 3: Same time, different status
                elif time[i] == time[j]:
                    if event[i] == 1 and event[j] == 0:
                        permissible += 1
                        if risk[i] > risk[j]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5
                    elif event[j] == 1 and event[i] == 0:
                        permissible += 1
                        if risk[j] > risk[i]:
                            concordant += 1
                        elif risk[i] == risk[j]:
                            concordant += 0.5
        
        if permissible == 0:
            return 0.5
        
        return concordant / permissible
    
    def predict(self, graphs: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for a set of graphs.
        
        Returns:
            risks: Predicted risk scores
            times: Survival times
            events: Event indicators
        """
        self.model.eval()
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            batch = Batch.from_data_list(graphs).to(self.config.DEVICE)
            risk = self.model(batch)
            
            all_risks = risk.cpu().numpy().flatten()
            all_times = batch.survival_time.cpu().numpy()
            all_events = batch.event.cpu().numpy()
        
        return all_risks, all_times, all_events
