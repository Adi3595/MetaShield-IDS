"""
MetaShield IDS — Main Training & Evaluation Pipeline
======================================================
Few-Shot Cybersecurity Threat Classification Using Meta-Learning

This script orchestrates:
1. CIC-IDS2017 data loading and preprocessing
2. Base/Novel class splitting for FSCIL
3. Episodic task generation
4. Prototypical Network training
5. Comprehensive evaluation (accuracy, baselines, plots)
6. Results export for research paper

Usage:
    python main.py                  # Full pipeline
    python main.py --eval-only      # Evaluation only (requires trained model)
    python main.py --quick          # Quick training (fewer epochs/tasks)
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch

from src.utils.helpers import set_seed, get_device, setup_logging, load_config, save_experiment_info
from src.data.cicids_loader import CICIDS2017Loader
from src.data.preprocessor import EpisodicTaskSampler
from src.models.prototypical_network import PrototypicalNetwork
from src.training.trainer import MetaTrainer
from src.evaluation.analyzer import FewShotEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='MetaShield IDS Training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only (load existing model)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer epochs and tasks')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable data caching')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    log_file = setup_logging()
    config = load_config(args.config)
    
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    device = get_device()
    
    results_dir = config.get('experiment', {}).get('results_dir', 'results')
    checkpoint_dir = config.get('experiment', {}).get('checkpoint_dir', 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Quick mode overrides
    if args.quick:
        config['training']['num_epochs'] = 10
        config['data']['num_train_tasks'] = 200
        config['data']['num_val_tasks'] = 50
        config['data']['num_test_tasks'] = 100
        logging.info("⚡ Quick mode enabled")
    
    logging.info("=" * 60)
    logging.info("🛡️  METASHIELD IDS")
    logging.info("   Few-Shot Cybersecurity Threat Classification")
    logging.info("   Using Meta-Learning (Prototypical Networks)")
    logging.info("=" * 60)
    
    # ===================== Phase 1: Data Loading =====================
    logging.info("\n📁 PHASE 1: Loading CIC-IDS2017 Dataset")
    logging.info("-" * 50)
    
    data_config = config.get('data', {})
    loader = CICIDS2017Loader(
        data_dir=data_config.get('data_dir', 'data/archive'),
        cache_dir=data_config.get('cache_dir', 'data/processed'),
        max_samples_per_class=data_config.get('max_samples_per_class', 50000),
        random_state=seed
    )
    
    X_train, X_test, y_train, y_test, class_names = loader.load_and_preprocess(
        use_cache=not args.no_cache
    )
    
    n_features = X_train.shape[1]
    n_classes = len(class_names)
    
    logging.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    logging.info(f"  Features: {n_features}, Classes: {n_classes}")
    logging.info(f"  Classes: {class_names}")
    
    # Class distribution
    for i, name in enumerate(class_names):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        logging.info(f"    {name}: {train_count} train, {test_count} test")
    
    # Base/Novel split
    base_data, novel_data = loader.get_base_novel_split(
        X_train, y_train, X_test, y_test
    )
    
    # ===================== Phase 2: Task Generation =====================
    logging.info("\n🔄 PHASE 2: Generating Episodic Tasks")
    logging.info("-" * 50)
    
    n_way = min(data_config.get('n_way', 5), len(base_data['classes']))
    k_shot = data_config.get('k_shot', 5)
    query_size = data_config.get('query_size', 15)
    
    # Training tasks (from base classes)
    train_sampler = EpisodicTaskSampler(
        base_data['X_train'], base_data['y_train'],
        n_way=n_way, k_shot=k_shot, query_size=query_size,
        num_tasks=data_config.get('num_train_tasks', 1000),
        random_state=seed
    )
    train_tasks = train_sampler.generate_tasks()
    
    # Validation tasks
    val_sampler = EpisodicTaskSampler(
        base_data['X_test'], base_data['y_test'],
        n_way=n_way, k_shot=k_shot, query_size=query_size,
        num_tasks=data_config.get('num_val_tasks', 200),
        random_state=seed + 1
    )
    val_tasks = val_sampler.generate_tasks()
    
    # Test tasks (from all classes)
    test_n_way = min(n_way, n_classes)
    test_sampler = EpisodicTaskSampler(
        X_test, y_test,
        n_way=test_n_way, k_shot=k_shot, query_size=query_size,
        num_tasks=data_config.get('num_test_tasks', 500),
        random_state=seed + 2
    )
    test_tasks = test_sampler.generate_tasks()
    
    logging.info(f"  Train tasks: {len(train_tasks)}")
    logging.info(f"  Val tasks: {len(val_tasks)}")
    logging.info(f"  Test tasks: {len(test_tasks)}")
    
    # ===================== Phase 3: Model Creation =====================
    logging.info("\n🧠 PHASE 3: Initializing Prototypical Network")
    logging.info("-" * 50)
    
    model_config = config.get('model', {})
    model = PrototypicalNetwork(
        input_dim=n_features,
        hidden_dim=model_config.get('hidden_dim', 256),
        embedding_dim=model_config.get('embedding_dim', 128),
        distance=model_config.get('distance', 'euclidean'),
        dropout=model_config.get('dropout', 0.3)
    )
    
    logging.info(f"  Parameters: {model.get_param_count():,}")
    logging.info(f"  Architecture: {model_config.get('hidden_dim', 256)}→{model_config.get('embedding_dim', 128)}")
    
    # ===================== Phase 4: Training =====================
    if not args.eval_only:
        logging.info("\n🏃 PHASE 4: Meta-Training")
        logging.info("-" * 50)
        
        training_config = config.get('training', {})
        trainer = MetaTrainer(
            model, training_config, 
            device=str(device), 
            checkpoint_dir=checkpoint_dir
        )
        
        history = trainer.train(
            train_tasks, val_tasks,
            num_epochs=training_config.get('num_epochs', 50)
        )
    else:
        # Load existing model
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            logging.error(f"No model found at {model_path}. Run training first.")
            sys.exit(1)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        logging.info(f"Loaded model from {model_path}")
        
        # Load history if available
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        if os.path.exists(history_path):
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = None
    
    # ===================== Phase 5: Evaluation =====================
    logging.info("\n📊 PHASE 5: Comprehensive Evaluation")
    logging.info("-" * 50)
    
    evaluator = FewShotEvaluator(model, device=str(device), results_dir=results_dir)
    
    all_results = evaluator.run_full_evaluation(
        X_train=base_data['X_train'],
        y_train=base_data['y_train'],
        X_test=X_test,
        y_test=y_test,
        test_tasks=test_tasks,
        class_names=class_names,
        training_history=history if not args.eval_only else history
    )
    
    # Save experiment info
    save_experiment_info(
        results_dir, config, model,
        data_info={
            'dataset': 'CIC-IDS2017',
            'n_features': n_features,
            'n_classes': n_classes,
            'class_names': class_names,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'base_classes': base_data['classes'],
            'novel_classes': novel_data['classes']
        }
    )
    
    # ===================== Summary =====================
    logging.info("\n" + "=" * 60)
    logging.info("✅ PIPELINE COMPLETE")
    logging.info("=" * 60)
    logging.info(f"\nOutputs:")
    logging.info(f"  📁 Model checkpoints: {checkpoint_dir}/")
    logging.info(f"  📊 Results & plots: {results_dir}/")
    logging.info(f"  📝 Log file: {log_file}")
    logging.info(f"\nKey Results:")
    
    fs = all_results.get('few_shot', {})
    acc = fs.get('accuracy', {})
    logging.info(f"  Accuracy: {acc.get('mean', 0):.4f} ± {acc.get('ci_95', 0):.4f}")
    logging.info(f"  F1-Score: {fs.get('macro_metrics', {}).get('f1_score', 0):.4f}")
    
    ks = all_results.get('k_shot_sweep', {})
    if ks:
        for k, v in sorted(ks.items()):
            logging.info(f"  {k}-shot: {v['mean']:.4f}")


if __name__ == "__main__":
    main()