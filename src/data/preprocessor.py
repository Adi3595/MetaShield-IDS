"""
Episodic Task Sampler for Few-Shot Learning
=============================================
Creates N-way K-shot episodic training/evaluation tasks
from the CIC-IDS2017 dataset for meta-learning.

Supports:
- Standard few-shot episodes
- Class-incremental episodes (base + novel)
- Configurable N-way, K-shot, query sizes
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class EpisodicTaskSampler:
    """
    Generates episodic few-shot tasks for Prototypical Network training.
    
    Each task consists of:
    - Support set: K examples per class (N*K total)
    - Query set: Q examples per class (N*Q total)
    """
    
    def __init__(self, X, y, n_way=5, k_shot=5, query_size=15,
                 num_tasks=1000, random_state=42):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_size: Number of query examples per class
            num_tasks: Number of episodes to generate
            random_state: Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.num_tasks = num_tasks
        self.rng = np.random.RandomState(random_state)
        
        # Build class-to-indices mapping
        self.class_indices = {}
        self.unique_classes = np.unique(y)
        
        for cls in self.unique_classes:
            self.class_indices[cls] = np.where(y == cls)[0]
        
        # Filter classes that have enough samples
        min_samples = k_shot + query_size
        self.valid_classes = [
            cls for cls in self.unique_classes
            if len(self.class_indices[cls]) >= min_samples
        ]
        
        if len(self.valid_classes) < n_way:
            logger.warning(
                f"Only {len(self.valid_classes)} classes have enough samples "
                f"(need {min_samples}), but n_way={n_way}. "
                f"Reducing n_way to {len(self.valid_classes)}."
            )
            self.n_way = len(self.valid_classes)
        
        logger.info(
            f"TaskSampler: {len(self.valid_classes)} valid classes, "
            f"{n_way}-way {k_shot}-shot, {query_size} queries, "
            f"{num_tasks} tasks"
        )
    
    def generate_tasks(self):
        """
        Generate all episodic tasks.
        
        Returns:
            List of task dicts, each containing:
                'support_x': (n_way * k_shot, n_features) tensor
                'support_y': (n_way * k_shot,) tensor
                'query_x': (n_way * query_size, n_features) tensor
                'query_y': (n_way * query_size,) tensor
                'class_map': mapping from episode labels to original labels
        """
        tasks = []
        
        for _ in range(self.num_tasks):
            task = self._sample_task()
            if task is not None:
                tasks.append(task)
        
        logger.info(f"Generated {len(tasks)} / {self.num_tasks} tasks")
        return tasks
    
    def _sample_task(self):
        """Sample a single N-way K-shot task."""
        # Select N random classes
        selected_classes = self.rng.choice(
            self.valid_classes, self.n_way, replace=False
        )
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        class_map = {}
        
        for episode_label, original_class in enumerate(selected_classes):
            class_map[episode_label] = original_class
            indices = self.class_indices[original_class]
            
            # Sample support + query indices
            total_needed = self.k_shot + self.query_size
            if len(indices) < total_needed:
                return None
            
            selected = self.rng.choice(indices, total_needed, replace=False)
            support_idx = selected[:self.k_shot]
            query_idx = selected[self.k_shot:]
            
            # Add to support set
            for idx in support_idx:
                support_x.append(self.X[idx])
                support_y.append(episode_label)
            
            # Add to query set
            for idx in query_idx:
                query_x.append(self.X[idx])
                query_y.append(episode_label)
        
        return {
            'support_x': torch.FloatTensor(np.array(support_x)),
            'support_y': torch.LongTensor(support_y),
            'query_x': torch.FloatTensor(np.array(query_x)),
            'query_y': torch.LongTensor(query_y),
            'class_map': class_map
        }
    
    def generate_incremental_tasks(self, novel_X, novel_y, 
                                    n_novel_way=2, n_base_way=3):
        """
        Generate class-incremental learning tasks.
        
        Each task includes:
        - n_base_way classes from base set (with full support)
        - n_novel_way classes from novel set (with few-shot support)
        
        This evaluates the model's ability to learn new classes
        without forgetting old ones.
        """
        # Novel class indices
        novel_classes = np.unique(novel_y)
        novel_class_indices = {}
        for cls in novel_classes:
            novel_class_indices[cls] = np.where(novel_y == cls)[0]
        
        valid_novel = [
            cls for cls in novel_classes
            if len(novel_class_indices[cls]) >= (self.k_shot + self.query_size)
        ]
        
        if len(valid_novel) < n_novel_way:
            n_novel_way = len(valid_novel)
        
        tasks = []
        for _ in range(min(self.num_tasks, 200)):
            # Sample base classes
            base_classes = self.rng.choice(
                self.valid_classes, min(n_base_way, len(self.valid_classes)), 
                replace=False
            )
            
            # Sample novel classes
            if len(valid_novel) > 0:
                novel_selected = self.rng.choice(
                    valid_novel, min(n_novel_way, len(valid_novel)), 
                    replace=False
                )
            else:
                novel_selected = []
            
            support_x, support_y = [], []
            query_x, query_y = [], []
            class_map = {}
            label_counter = 0
            
            # Add base class samples
            for original_class in base_classes:
                class_map[label_counter] = ('base', int(original_class))
                indices = self.class_indices[original_class]
                total_needed = self.k_shot + self.query_size
                selected = self.rng.choice(indices, total_needed, replace=False)
                
                for idx in selected[:self.k_shot]:
                    support_x.append(self.X[idx])
                    support_y.append(label_counter)
                for idx in selected[self.k_shot:]:
                    query_x.append(self.X[idx])
                    query_y.append(label_counter)
                
                label_counter += 1
            
            # Add novel class samples
            for original_class in novel_selected:
                class_map[label_counter] = ('novel', int(original_class))
                indices = novel_class_indices[original_class]
                total_needed = self.k_shot + self.query_size
                if len(indices) < total_needed:
                    continue
                selected = self.rng.choice(indices, total_needed, replace=False)
                
                for idx in selected[:self.k_shot]:
                    support_x.append(novel_X[idx])
                    support_y.append(label_counter)
                for idx in selected[self.k_shot:]:
                    query_x.append(novel_X[idx])
                    query_y.append(label_counter)
                
                label_counter += 1
            
            if len(support_x) > 0:
                tasks.append({
                    'support_x': torch.FloatTensor(np.array(support_x)),
                    'support_y': torch.LongTensor(support_y),
                    'query_x': torch.FloatTensor(np.array(query_x)),
                    'query_y': torch.LongTensor(query_y),
                    'class_map': class_map,
                    'is_incremental': True
                })
        
        logger.info(f"Generated {len(tasks)} incremental tasks")
        return tasks


class EpisodicDataset(Dataset):
    """PyTorch Dataset wrapper for pre-generated episodic tasks."""
    
    def __init__(self, tasks):
        self.tasks = tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]