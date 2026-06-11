"""
Utility helpers for MetaShield IDS
====================================
Reproducibility, logging, device management, and configuration utilities.
"""

import torch
import numpy as np
import random
import os
import logging
import yaml
import json
from datetime import datetime


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logging.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def setup_logging(log_dir='logs', level=logging.INFO):
    """Setup logging with both console and file handlers."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'metashield_{timestamp}.log')
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging to {log_file}")
    return log_file


def load_config(config_path='configs/training_config.yaml'):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_experiment_info(results_dir, config, model, data_info):
    """Save experiment metadata for reproducibility."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'model': {
            'type': model.__class__.__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'data': data_info,
        'system': {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        }
    }
    
    filepath = os.path.join(results_dir, 'experiment_info.json')
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    return info


def format_metrics_table(results):
    """Format results as a printable table."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'Value':<20} {'95% CI':<20}")
    lines.append("-" * 70)
    
    if 'accuracy' in results:
        acc = results['accuracy']
        lines.append(f"{'Accuracy':<30} {acc['mean']:.4f}{'':>12} ±{acc['ci_95']:.4f}")
    
    if 'macro_metrics' in results:
        m = results['macro_metrics']
        lines.append(f"{'Precision (macro)':<30} {m['precision']:.4f}")
        lines.append(f"{'Recall (macro)':<30} {m['recall']:.4f}")
        lines.append(f"{'F1-Score (macro)':<30} {m['f1_score']:.4f}")
    
    lines.append("=" * 70)
    return "\n".join(lines)
