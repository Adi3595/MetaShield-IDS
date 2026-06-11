"""
Meta-Training Pipeline for Few-Shot Intrusion Detection
=========================================================
Episodic training loop with:
- Learning rate scheduling (cosine annealing)
- Early stopping with patience
- Gradient clipping
- Comprehensive metric logging
- Model checkpointing
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetaTrainer:
    """
    Meta-learning trainer for Prototypical Networks.
    
    Implements episodic training with proper validation,
    learning rate scheduling, and early stopping.
    """
    
    def __init__(self, model, config, device='cuda', checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        
        # Early stopping
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 5.0)
        
        logger.info(
            f"MetaTrainer initialized: lr={config.get('learning_rate', 0.001)}, "
            f"patience={self.patience}, device={device}"
        )
    
    def train_epoch(self, train_tasks, epoch):
        """Train for one epoch over episodic tasks."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        n_tasks = len(train_tasks)
        start_time = time.time()
        
        # Shuffle tasks
        indices = np.random.permutation(n_tasks)
        
        for i, idx in enumerate(indices):
            task = train_tasks[idx]
            
            support_x = task['support_x'].to(self.device)
            support_y = task['support_y'].to(self.device)
            query_x = task['query_x'].to(self.device)
            query_y = task['query_y'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, acc = self.model.compute_loss(support_x, support_y, query_x, query_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            # Progress logging
            if (i + 1) % 50 == 0:
                avg_loss = total_loss / (i + 1)
                avg_acc = total_acc / (i + 1)
                logger.info(
                    f"  Epoch {epoch+1} | Task {i+1}/{n_tasks} | "
                    f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
                )
        
        avg_loss = total_loss / n_tasks
        avg_acc = total_acc / n_tasks
        epoch_time = time.time() - start_time
        
        return avg_loss, avg_acc, epoch_time
    
    def validate(self, val_tasks):
        """Validate on episodic tasks."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n_tasks = len(val_tasks)
        
        with torch.no_grad():
            for task in val_tasks:
                support_x = task['support_x'].to(self.device)
                support_y = task['support_y'].to(self.device)
                query_x = task['query_x'].to(self.device)
                query_y = task['query_y'].to(self.device)
                
                loss, acc = self.model.compute_loss(
                    support_x, support_y, query_x, query_y
                )
                
                total_loss += loss.item()
                total_acc += acc.item()
        
        avg_loss = total_loss / n_tasks
        avg_acc = total_acc / n_tasks
        
        return avg_loss, avg_acc
    
    def train(self, train_tasks, val_tasks, num_epochs=None):
        """
        Main training loop with early stopping and checkpointing.
        
        Args:
            train_tasks: List of training episodes
            val_tasks: List of validation episodes
            num_epochs: Override config epochs
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 50)
        
        logger.info("=" * 60)
        logger.info(f"Starting meta-training for {num_epochs} epochs")
        logger.info(f"  Train tasks: {len(train_tasks)}")
        logger.info(f"  Val tasks: {len(val_tasks)}")
        logger.info(f"  Model params: {self.model.get_param_count():,}")
        logger.info("=" * 60)
        
        total_start = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(
                train_tasks, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate(val_tasks)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                improved = True
                
                # Save best model
                self._save_checkpoint(epoch, 'best_model.pt', val_loss, val_acc)
                logger.info(f"  ★ New best model (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch+1} "
                        f"(patience={self.patience})"
                    )
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pt', val_loss, val_acc)
        
        # Save final model
        self._save_checkpoint(epoch, 'final_model.pt', val_loss, val_acc)
        
        total_time = time.time() - total_start
        
        # Save training history
        self._save_history()
        
        logger.info("=" * 60)
        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(f"Best validation: loss={self.best_val_loss:.4f}, acc={self.best_val_acc:.4f}")
        logger.info("=" * 60)
        
        return self.history
    
    def _save_checkpoint(self, epoch, filename, val_loss, val_acc):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config,
        }, filepath)
    
    def _save_history(self):
        """Save training history to JSON."""
        # Convert numpy types for JSON serialization
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [
                float(v) if isinstance(v, (np.floating, float)) else v 
                for v in values
            ]
        
        filepath = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        logger.info(f"Training history saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(
            f"Loaded checkpoint from {filepath} "
            f"(epoch={checkpoint.get('epoch', '?')}, "
            f"val_acc={checkpoint.get('val_acc', '?')})"
        )
        return checkpoint