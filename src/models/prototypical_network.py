"""
Enhanced Prototypical Network for Few-Shot Intrusion Detection
================================================================
Implements an attention-enhanced embedding network with:
- Multi-layer feature extraction with residual connections
- Self-attention mechanism for feature importance
- Configurable distance metrics (Euclidean, Cosine)
- Class-incremental prototype registry

Reference: Snell, J., Swersky, K., & Zemel, R. (2017).
"Prototypical Networks for Few-shot Learning." NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BayesianPrototypeAttention(nn.Module):
    """
    [BFPU] Bayesian Few-Shot with Predictive Uncertainty.
    Attention mechanism to weight support examples and compute Bayesian prototypes (mean + covariance).
    """
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        # Learnable minimum variance to prevent collapse
        self.min_var = nn.Parameter(torch.ones(dim) * 1e-4)
        
    def forward(self, support_embeddings):
        # support_embeddings: (n_support, dim)
        if support_embeddings.shape[0] == 1:
            mu = support_embeddings.squeeze(0)
            var = F.softplus(self.min_var)
            return mu, var
            
        attn_weights = self.attention(support_embeddings) # (n_support, 1)
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # Weighted mean (μ)
        mu = (support_embeddings * attn_weights).sum(dim=0)
        
        # Weighted variance (Σ) - represents epistemic uncertainty
        diff = support_embeddings - mu
        var = (attn_weights * (diff ** 2)).sum(dim=0) + F.softplus(self.min_var)
        
        return mu, var


class SelfAttention(nn.Module):
    """Self-attention layer for learning feature importance."""
    
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
    
    def forward(self, x):
        # x: (batch, dim)
        q = self.query(x).unsqueeze(1)  # (batch, 1, dim)
        k = self.key(x).unsqueeze(1)    # (batch, 1, dim)
        v = self.value(x).unsqueeze(1)  # (batch, 1, dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(1)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class AttentionEmbeddingNetwork(nn.Module):
    """
    Enhanced embedding network with attention mechanism.
    
    Architecture:
    1. Input projection layer
    2. Two residual blocks for deep feature extraction
    3. Self-attention for feature importance weighting
    4. Final embedding projection
    """
    
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Deep feature extraction with residual connections
        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)
        
        # Self-attention
        self.attention = SelfAttention(hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Residual feature extraction
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Self-attention
        attn_out = self.attention(x)
        x = self.attn_norm(x + attn_out)
        
        # Final embedding
        embedding = self.embedding_proj(x)
        
        # L2 normalize embeddings for better distance computation
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Intrusion Detection.
    
    Computes class prototypes from support examples and classifies
    query examples based on distance to prototypes.
    
    Supports:
    - Euclidean distance
    - Cosine similarity
    - Class-incremental prototype registry
    """
    
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, 
                 distance='bayesian', dropout=0.3):
        super().__init__()
        
        self.embedding_net = AttentionEmbeddingNetwork(
            input_dim, hidden_dim, embedding_dim, dropout
        )
        self.distance_type = distance
        self.embedding_dim = embedding_dim
        
        # [BFPU] Bayesian prototype generation
        self.proto_attention = BayesianPrototypeAttention(embedding_dim)
        
        # Learnable distance blending (if hybrid)
        if distance == 'hybrid':
            self.dist_alpha = nn.Parameter(torch.tensor(0.0))  # 0.0 means sigmoid(0) = 0.5
            
        # Learnable temperature for scaling logits
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Prototype registry for incremental learning (Stores mu and var)
        self.prototype_registry = {}
        
        logger.info(
            f"PrototypicalNetwork: input={input_dim}, hidden={hidden_dim}, "
            f"embed={embedding_dim}, distance={distance}"
        )
    
    def forward(self, support_x, support_y, query_x):
        """
        Forward pass for episodic training/evaluation.
        
        Args:
            support_x: (n_support, n_features) support examples
            support_y: (n_support,) support labels  
            query_x: (n_query, n_features) query examples
            
        Returns:
            logits: (n_query, n_way) classification logits
        """
        # Compute embeddings
        support_embeddings = self.embedding_net(support_x)
        query_embeddings = self.embedding_net(query_x)
        
        # Compute class prototypes
        unique_classes = torch.unique(support_y)
        n_way = len(unique_classes)
        prototypes_mu = []
        prototypes_var = []
        
        for cls in unique_classes:
            mask = (support_y == cls)
            class_embeddings = support_embeddings[mask]
            
            # [BFPU] Robust Bayesian Prototype Generation
            mu, var = self.proto_attention(class_embeddings)
            prototypes_mu.append(mu)
            prototypes_var.append(var)
        
        prototypes_mu = torch.stack(prototypes_mu)  # (n_way, embedding_dim)
        prototypes_var = torch.stack(prototypes_var)  # (n_way, embedding_dim)
        
        # Compute distances/similarities
        logits = self._compute_logits(query_embeddings, prototypes_mu, prototypes_var)
        
        return logits
    
    def _compute_logits(self, query_embeddings, prototypes_mu, prototypes_var=None):
        """
        Compute classification logits based on distance metric.
        
        Args:
            query_embeddings: (n_query, embedding_dim)
            prototypes_mu: (n_way, embedding_dim)
            prototypes_var: (n_way, embedding_dim) Optional, for Bayesian
            
        Returns:
            logits: (n_query, n_way)
        """
        n_query = query_embeddings.shape[0]
        n_way = prototypes_mu.shape[0]
        
        if self.distance_type == 'bayesian' and prototypes_var is not None:
            # [BFPU] Expected Log-Likelihood of query under Gaussian prototype
            query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)
            mu_expanded = prototypes_mu.unsqueeze(0).expand(n_query, -1, -1)
            var_expanded = prototypes_var.unsqueeze(0).expand(n_query, -1, -1)
            
            # Mahalanobis-like distance factoring in uncertainty
            distances = 0.5 * torch.sum(
                torch.log(var_expanded) + (query_expanded - mu_expanded) ** 2 / var_expanded, 
                dim=2
            )
            logits = -distances * self.temperature
            
        elif self.distance_type == 'cosine':
            # Cosine similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=-1)
            proto_norm = F.normalize(prototypes_mu, p=2, dim=-1)
            logits = torch.mm(query_norm, proto_norm.t()) * self.temperature
            
        elif self.distance_type == 'euclidean':
            # Negative Euclidean distance
            query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)
            proto_expanded = prototypes_mu.unsqueeze(0).expand(n_query, -1, -1)
            
            distances = torch.pow(query_expanded - proto_expanded, 2).sum(dim=2)
            logits = -distances * self.temperature
            
        elif self.distance_type == 'hybrid':
            # Learnable Hybrid Distance
            query_norm = F.normalize(query_embeddings, p=2, dim=-1)
            proto_norm = F.normalize(prototypes_mu, p=2, dim=-1)
            cosine_sim = torch.mm(query_norm, proto_norm.t())
            
            query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)
            proto_expanded = prototypes_mu.unsqueeze(0).expand(n_query, -1, -1)
            euc_dist = torch.pow(query_expanded - proto_expanded, 2).sum(dim=2)
            
            alpha = torch.sigmoid(self.dist_alpha)
            
            # Combine similarities
            logits = (alpha * cosine_sim) - ((1.0 - alpha) * euc_dist)
            logits = logits * self.temperature
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return logits
    
    def compute_loss(self, support_x, support_y, query_x, query_y, apply_arcl=True):
        """
        Compute loss for episodic training with optional [ARCL] Hard Negative Mining.
        """
        logits = self.forward(support_x, support_y, query_x)
        
        # Standard cross entropy
        ce_loss = F.cross_entropy(logits, query_y, reduction='none')
        
        if apply_arcl:
            # [ARCL] Adversarial Hard Negative Mining
            # Weight the hardest examples (top 20% highest losses) 3x more
            loss_weights = torch.ones_like(ce_loss)
            k_hard = max(1, int(0.2 * ce_loss.shape[0]))
            
            _, hard_indices = torch.topk(ce_loss, k_hard)
            loss_weights[hard_indices] = 3.0
            
            loss = (ce_loss * loss_weights).mean()
        else:
            loss = ce_loss.mean()
        
        # Compute accuracy for logging
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_y).float().mean()
        
        return loss, acc
    
    def get_embeddings(self, x):
        """Get embeddings for input features (inference mode)."""
        self.eval()
        with torch.no_grad():
            embeddings = self.embedding_net(x)
        return embeddings
    
    def register_prototype(self, class_name, examples):
        """
        Register a new class prototype for incremental learning.
        
        Args:
            class_name: Name of the new attack class
            examples: Tensor of shape (n_examples, n_features)
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.embedding_net(examples)
            mu, var = self.proto_attention(embeddings)
        
        self.prototype_registry[class_name] = {'mu': mu, 'var': var}
        logger.info(f"Registered Bayesian prototype for '{class_name}' from {len(examples)} examples")
        return mu
    
    def detect_with_registry(self, x, threshold=0.5):
        """
        Detect threats using the registered prototype registry.
        
        Args:
            x: Input features tensor (1, n_features) or (n_features,)
            threshold: Confidence threshold
            
        Returns:
            (class_name, confidence, uncertainty) or (None, max_confidence, uncertainty)
        """
        if not self.prototype_registry:
            return None, 0.0, 0.0
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            embedding = self.embedding_net(x)
        
        best_match = None
        best_confidence = 0.0
        best_uncertainty = 1.0
        
        for class_name, proto_dict in self.prototype_registry.items():
            mu = proto_dict['mu']
            var = proto_dict.get('var', torch.ones_like(mu))
            
            if self.distance_type == 'bayesian':
                # [BFPU] Full Bayesian distance: Mahalanobis + log-variance penalty
                # Use sum (not mean) so distance scales with embedding dimensionality,
                # preventing L2-normalized embeddings from collapsing distances to ~0.
                diff = embedding.squeeze() - mu
                mahal_dist = 0.5 * torch.sum((diff ** 2) / var).item()
                log_var_penalty = 0.5 * torch.sum(torch.log(var)).item()
                distance = mahal_dist + log_var_penalty
                
                # Exponential confidence with temperature for meaningful dynamic range
                embed_dim = embedding.shape[-1]
                temperature = embed_dim * 0.25
                import math
                confidence = math.exp(-max(0, distance) / temperature)
                confidence = max(0.0, min(1.0, confidence))
                uncertainty = torch.mean(var).item()
            elif self.distance_type == 'cosine':
                similarity = F.cosine_similarity(
                    embedding, mu.unsqueeze(0)
                ).item()
                confidence = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]
                uncertainty = 0.0
            else:
                distance = torch.dist(embedding, mu).item()
                confidence = 1.0 / (1.0 + distance)
                uncertainty = 0.0
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = class_name
                best_uncertainty = uncertainty
        
        if best_confidence >= threshold:
            return best_match, best_confidence, best_uncertainty
        return None, best_confidence, best_uncertainty
    
    def get_param_count(self):
        """Get total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)