"""
MetaShield Threat Detector — Production Module
=================================================
Real-time few-shot threat detection using trained Prototypical Network.
Supports:
- Loading trained models
- Learning new attack types from few examples
- Real-time network flow classification
- Confidence-based threat scoring
"""

import torch
import numpy as np
import time
import os
import json
import pickle
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class FewShotThreatDetector:
    """
    Production-ready few-shot threat detector.
    
    Uses a trained Prototypical Network to detect known threats
    and learn new attack patterns from just a few examples.
    """
    
    def __init__(self, model_path='checkpoints/best_model.pt',
                 config_path='configs/training_config.yaml',
                 data_cache_path='data/processed/cicids2017_processed.pkl'):
        
        logger.info("🛡️ Initializing MetaShield Threat Detector...")
        
        # Load config
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'model': {'input_dim': 78, 'hidden_dim': 256, 
                                     'embedding_dim': 128, 'distance': 'euclidean'}}
        
        # Determine input dimension
        self.input_dim = self.config['model'].get('input_dim', 78)
        
        # Load scaler for feature normalization
        self.scaler = None
        if os.path.exists(data_cache_path):
            try:
                with open(data_cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self.scaler = cached.get('scaler')
                self.input_dim = cached.get('n_features', self.input_dim)
                self.class_names_orig = cached.get('class_names', [])
                logger.info(f"  Loaded scaler ({self.input_dim} features)")
            except Exception as e:
                logger.warning(f"  Could not load data cache: {e}")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        from src.models.prototypical_network import PrototypicalNetwork
        model_config = self.config.get('model', {})
        self.model = PrototypicalNetwork(
            input_dim=self.input_dim,
            hidden_dim=model_config.get('hidden_dim', 256),
            embedding_dim=model_config.get('embedding_dim', 128),
            distance=model_config.get('distance', 'euclidean'),
            dropout=0.0  # No dropout at inference
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"  Loaded model from {model_path}")
        else:
            logger.warning(f"  No model found at {model_path}, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Attack registry
        self.known_attacks = {}  # name -> prototype tensor
        self.registered_examples = {}  # name -> list of raw examples for serialization
        self.attack_history = deque(maxlen=1000)
        self.threshold = 0.5
        self.total_flows_analyzed = 0
        self.total_threats_detected = 0
        self.start_time = time.time()
        
        logger.info(f"✅ Detector ready on {self.device}")
        logger.info(f"   Input features: {self.input_dim}")
        logger.info(f"   Embedding dim: {model_config.get('embedding_dim', 128)}")
    
    def learn_new_attack(self, attack_name, examples):
        """
        Learn a new attack type from few examples.
        
        Args:
            attack_name: Name of the attack (e.g., "DDoS", "BruteForce")
            examples: List of feature vectors (each of length input_dim)
                      Can be raw lists or numpy arrays.
        
        Returns:
            prototype: The computed prototype vector
        """
        logger.info(f"📚 Learning new attack: '{attack_name}' from {len(examples)} examples")
        
        # Convert to numpy and normalize if scaler is available and features are raw
        examples_array = np.array(examples, dtype=np.float32)
        
        if self.scaler is not None and examples_array.size > 0:
            try:
                # Heuristic: if maximum absolute value of features is large (>= 25.0),
                # we assume features are raw (e.g. destination ports, durations) and apply scaler.
                # If they are small (< 25.0), they are likely already pre-scaled/normalized.
                if np.max(np.abs(examples_array)) >= 25.0:
                    examples_array = self.scaler.transform(examples_array)
                else:
                    logger.info("  Input examples are already normalized (max abs < 25.0); skipping scaler.transform")
            except Exception as e:
                logger.warning(f"  Scaler transformation failed: {e}")
        
        # Compute prototype
        support_x = torch.from_numpy(examples_array).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.embedding_net(support_x)
            # [BFPU] Bayesian Prototype Generation
            mu, var = self.model.proto_attention(embeddings)
        
        # Store prototypes in float32 for accurate distance computation
        prototype = {'mu': mu.detach().clone(), 'var': var.detach().clone()}
        self.known_attacks[attack_name] = prototype
        self.registered_examples[attack_name] = [list(e) if isinstance(e, (list, np.ndarray)) else e for e in examples]
        logger.info(f"✅ Registered '{attack_name}' (Bayesian Prototype)")
        
        return prototype
    
    def detect(self, network_flow):
        """
        Analyze a single network flow for threats.
        
        Args:
            network_flow: Feature vector (list or numpy array of length input_dim)
        
        Returns:
            (attack_name, confidence) or (None, max_confidence)
        """
        self.total_flows_analyzed += 1
        
        if not self.known_attacks:
            return None, 0.0
        
        # Prepare input
        flow_array = np.array([network_flow], dtype=np.float32)
        
        if self.scaler is not None and flow_array.size > 0:
            try:
                # Heuristic: if maximum absolute value of features is large (>= 25.0),
                # we assume features are raw and apply scaler.
                if np.max(np.abs(flow_array)) >= 25.0:
                    flow_array = self.scaler.transform(flow_array)
                else:
                    logger.debug("  Input flow is already normalized (max abs < 25.0); skipping scaler.transform")
            except Exception as e:
                logger.warning(f"  Scaler transformation failed: {e}")
        
        x = torch.from_numpy(flow_array).to(self.device)
        
        # Get embedding in float32 for accurate distance computation
        with torch.no_grad():
            embedding = self.model.embedding_net(x)
        
        # Compare with all registered attacks
        best_match = None
        best_confidence = 0.0
        best_uncertainty = 1.0
        
        # Embedding dimension for scaling
        embed_dim = embedding.shape[-1]
        
        for attack_name, proto_dict in self.known_attacks.items():
            mu = proto_dict['mu']
            var = proto_dict['var']
            
            # [BFPU] Bayesian distance: Mahalanobis-like with uncertainty
            # Use sum (not mean) so distance scales with embedding dimensionality,
            # preventing L2-normalized embeddings from collapsing distances to ~0.
            # The log(var) term penalizes high-uncertainty prototypes.
            diff = embedding.squeeze() - mu
            mahal_dist = 0.5 * torch.sum((diff ** 2) / var).item()
            log_var_penalty = 0.5 * torch.sum(torch.log(var)).item()
            distance = mahal_dist + log_var_penalty
            
            # Exponential confidence with temperature for better dynamic range.
            # With L2-normalized 128-dim embeddings, typical distances range 20-80.
            # Temperature controls how quickly confidence drops with distance.
            temperature = embed_dim * 0.25  # Scale with dimensionality
            confidence = np.exp(-max(0, distance) / temperature)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            
            uncertainty = torch.mean(var).item()
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = attack_name
                best_uncertainty = uncertainty
        
        # Apply threshold
        if best_confidence >= self.threshold:
            self.total_threats_detected += 1
            
            # Log alert
            self.attack_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'attack': best_match,
                'confidence': round(best_confidence, 4),
                'uncertainty': round(best_uncertainty, 4), # [BFPU] Log uncertainty
                'severity': self._get_severity(best_confidence)
            })
            
            return best_match, best_confidence
        
        return None, best_confidence
    
    def detect_batch(self, flows):
        """
        Analyze a batch of network flows.
        
        Args:
            flows: List of feature vectors or numpy array (n_flows, n_features)
            
        Returns:
            List of (attack_name, confidence) tuples
        """
        results = []
        for flow in flows:
            attack, confidence = self.detect(flow)
            results.append((attack, confidence))
        return results
    
    def forget_attack(self, attack_name):
        """Remove an attack from the registry."""
        deleted = False
        if attack_name in self.known_attacks:
            del self.known_attacks[attack_name]
            deleted = True
        if attack_name in self.registered_examples:
            del self.registered_examples[attack_name]
            deleted = True
        if deleted:
            logger.info(f"Forgot attack: '{attack_name}'")
            return True
        return False
    
    def get_stats(self):
        """Get detector statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'known_attacks': list(self.known_attacks.keys()),
            'total_flows_analyzed': self.total_flows_analyzed,
            'total_threats_detected': self.total_threats_detected,
            'detection_rate': (self.total_threats_detected / max(1, self.total_flows_analyzed)),
            'recent_alerts': list(self.attack_history)[-50:],
            'total_alerts': len(self.attack_history),
            'uptime_seconds': uptime,
            'threshold': self.threshold,
            'model_info': {
                'type': 'Prototypical Network (Attention-Enhanced)',
                'device': str(self.device),
                'input_dim': self.input_dim,
                'embedding_dim': self.config['model'].get('embedding_dim', 128)
            }
        }
    
    def _get_severity(self, confidence):
        """Map confidence to severity level."""
        if confidence > 0.9:
            return 'critical'
        elif confidence > 0.7:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        return 'low'
    
    def load_attack_signatures(self, filepath='attack_signatures.json'):
        """Load pre-defined attack signatures from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                signatures = json.load(f)
            for name, examples in signatures.items():
                self.learn_new_attack(name, examples)
            logger.info(f"Loaded {len(signatures)} attack signatures from {filepath}")
    
    def save_attack_signatures(self, filepath='attack_signatures.json'):
        """Save current attack prototypes."""
        with open(filepath, 'w') as f:
            json.dump(self.registered_examples, f, indent=2)
        logger.info(f"Saved {len(self.registered_examples)} attack signatures to {filepath}")