#!/usr/bin/env python3
"""
Early stopping utility for training
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: How many epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda x, y: x < y - self.min_delta
        elif mode == 'max':
            self.is_better = lambda x, y: x > y + self.min_delta
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'min' or 'max'")
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training
        
        Args:
            score: Current score (loss or metric)
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"EarlyStopping: Initial score = {score:.6f}")
            return False
        
        if self.is_better(score, self.best_score):
            # Improvement found
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Score improved to {score:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training!")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    @property
    def has_improved(self) -> bool:
        """Check if score has improved in the last call"""
        return self.counter == 0 and self.best_score is not None


class ModelCheckpoint:
    """
    Save model when monitored metric improves
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Save only when metric improves
            verbose: Print messages
        """
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def check(self, score: float) -> bool:
        """
        Check if should save model
        
        Args:
            score: Current score
        
        Returns:
            True if should save, False otherwise
        """
        if not self.save_best_only:
            return True
        
        if self.monitor_op(score, self.best):
            if self.verbose:
                print(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {score:.6f}")
            self.best = score
            return True
        
        return False


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric has stopped improving
    """
    
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            factor: Factor to reduce LR by
            patience: Epochs to wait before reducing
            min_lr: Minimum learning rate
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        
        self.best = None
        self.counter = 0
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y
        elif mode == 'max':
            self.is_better = lambda x, y: x > y
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def check(self, score: float, current_lr: float) -> float:
        """
        Check if should reduce learning rate
        
        Args:
            score: Current score
            current_lr: Current learning rate
        
        Returns:
            New learning rate
        """
        if self.best is None:
            self.best = score
            return current_lr
        
        if self.is_better(score, self.best):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr:
                    if self.verbose:
                        print(f"ReduceLROnPlateau: Reducing LR from {current_lr:.2e} to {new_lr:.2e}")
                    self.counter = 0
                    return new_lr
        
        return current_lr


if __name__ == "__main__":
    # Test early stopping
    print("Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    scores = [0.5, 0.4, 0.39, 0.391, 0.392, 0.393]
    for epoch, score in enumerate(scores):
        print(f"\nEpoch {epoch}: score = {score}")
        should_stop = early_stopping(score)
        if should_stop:
            print("Stopping training!")
            break
    
    # Test model checkpoint
    print("\n\nTesting ModelCheckpoint...")
    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min')
    
    scores = [0.5, 0.4, 0.45, 0.35, 0.36]
    for epoch, score in enumerate(scores):
        should_save = checkpoint.check(score)
        print(f"Epoch {epoch}: score = {score}, save = {should_save}")
    
    # Test reduce LR on plateau
    print("\n\nTesting ReduceLROnPlateau...")
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=2)
    
    current_lr = 0.001
    scores = [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4]
    for epoch, score in enumerate(scores):
        new_lr = lr_scheduler.check(score, current_lr)
        print(f"Epoch {epoch}: score = {score}, lr = {current_lr:.1e} -> {new_lr:.1e}")
        current_lr = new_lr