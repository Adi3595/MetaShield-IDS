import torch
import torch.nn as nn
import torch.optim as optim
import copy

class MAMLModel(nn.Module):
    """Model-Agnostic Meta-Learning implementation"""
    def __init__(self, input_dim=78, hidden_dim=128, output_dim=5):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    
    def adapt(self, support_x, support_y, steps=5):
        """Adapt to new task"""
        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(steps):
            pred = adapted_model(support_x)
            loss = nn.functional.cross_entropy(pred, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_step(self, tasks):
        """Meta-update across multiple tasks"""
        meta_loss = 0
        
        for task in tasks:
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            
            adapted = self.adapt(support_x, support_y)
            pred = adapted(query_x)
            loss = nn.functional.cross_entropy(pred, query_y)
            meta_loss += loss
        
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()
        
        return meta_loss.item()