"""
Comprehensive Evaluation & Analysis for Few-Shot IDS
======================================================
Research-grade evaluation framework with:
- N-way K-shot accuracy measurement
- Per-class precision, recall, F1-score
- Confusion matrix generation
- t-SNE embedding visualization
- Class-incremental learning evaluation
- Baseline comparison (SVM, Random Forest, KNN)
- Statistical significance (confidence intervals)
- Publication-quality plot generation
"""

import torch
import numpy as np
import os
import json
import logging
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)


class FewShotEvaluator:
    """
    Comprehensive evaluator for few-shot intrusion detection.
    
    Generates all metrics and visualizations needed for a research paper.
    """
    
    def __init__(self, model, device='cuda', results_dir='results'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    def evaluate_few_shot(self, test_tasks, class_names=None):
        """
        Evaluate few-shot classification over multiple episodes.
        
        Args:
            test_tasks: List of episodic tasks
            class_names: Optional list of class names
            
        Returns:
            Dict with accuracy stats, per-class metrics, confusion matrix
        """
        logger.info(f"Evaluating on {len(test_tasks)} episodes...")
        
        all_preds = []
        all_labels = []
        episode_accuracies = []
        
        with torch.no_grad():
            for task in test_tasks:
                support_x = task['support_x'].to(self.device)
                support_y = task['support_y'].to(self.device)
                query_x = task['query_x'].to(self.device)
                query_y = task['query_y'].to(self.device)
                
                logits = self.model(support_x, support_y, query_x)
                preds = torch.argmax(logits, dim=1)
                
                acc = (preds == query_y).float().mean().item()
                episode_accuracies.append(acc)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(query_y.cpu().numpy().tolist())
        
        # Compute statistics
        episode_accs = np.array(episode_accuracies)
        mean_acc = np.mean(episode_accs)
        std_acc = np.std(episode_accs)
        ci_95 = 1.96 * std_acc / np.sqrt(len(episode_accs))
        
        # Per-class metrics
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        
        unique_labels = np.unique(np.concatenate([all_preds_np, all_labels_np]))
        
        precision = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        
        # Per-class breakdown
        if class_names is not None:
            all_class_labels = list(range(len(class_names)))
        else:
            all_class_labels = sorted(
                np.unique(np.concatenate([all_labels_np, all_preds_np]))
            )

        per_class_precision = precision_score(
            all_labels_np,
            all_preds_np,
            labels=all_class_labels,
            average=None,
            zero_division=0
        )

        per_class_recall = recall_score(
            all_labels_np,
            all_preds_np,
            labels=all_class_labels,
            average=None,
            zero_division=0
        )

        per_class_f1 = f1_score(
            all_labels_np,
            all_preds_np,
            labels=all_class_labels,
            average=None,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(
            all_labels_np,
            all_preds_np,
            labels=all_class_labels
        )
        
        results = {
            'accuracy': {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'ci_95': float(ci_95),
                'min': float(np.min(episode_accs)),
                'max': float(np.max(episode_accs)),
                'n_episodes': len(episode_accs)
            },
            'macro_metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist()
            },
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Few-Shot Results: Acc={mean_acc:.4f}±{ci_95:.4f} "
                     f"(95% CI), P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        
        return results
    
    def evaluate_k_shot_sweep(self, X, y, k_shots=[1, 3, 5, 10], 
                               n_way=5, query_size=15, n_episodes=200):
        """
        Evaluate accuracy across different K-shot values.
        Important for showing few-shot learning capability in the paper.
        """
        from src.data.preprocessor import EpisodicTaskSampler
        
        results = {}
        
        for k in k_shots:
            logger.info(f"Evaluating {n_way}-way {k}-shot...")
            
            sampler = EpisodicTaskSampler(
                X, y, n_way=n_way, k_shot=k, 
                query_size=query_size, num_tasks=n_episodes,
                random_state=42
            )
            tasks = sampler.generate_tasks()
            
            episode_accs = []
            with torch.no_grad():
                for task in tasks:
                    support_x = task['support_x'].to(self.device)
                    support_y = task['support_y'].to(self.device)
                    query_x = task['query_x'].to(self.device)
                    query_y = task['query_y'].to(self.device)
                    
                    logits = self.model(support_x, support_y, query_x)
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == query_y).float().mean().item()
                    episode_accs.append(acc)
            
            accs = np.array(episode_accs)
            results[k] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'ci_95': float(1.96 * np.std(accs) / np.sqrt(len(accs)))
            }
            
            logger.info(f"  {k}-shot: {np.mean(accs):.4f}±{1.96*np.std(accs)/np.sqrt(len(accs)):.4f}")
        
        return results
    
    def evaluate_baselines(self, X_train, y_train, X_test, y_test, 
                           k_shots=[1, 5, 10]):
        """
        Compare against traditional ML baselines using the same K-shot setup.
        Critical for research paper - shows meta-learning advantage.
        """
        logger.info("Evaluating baseline methods...")
        
        baselines = {
            'SVM': SVC(kernel='rbf', C=1.0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=3)
        }
        
        results = {}
        
        for k in k_shots:
            results[k] = {}
            
            for name, clf in baselines.items():
                accuracies = []
                n_trials = 50
                
                for trial in range(n_trials):
                    rng = np.random.RandomState(trial)
                    
                    # Sample k examples per class (few-shot setup)
                    unique_classes = np.unique(y_train)
                    few_shot_X, few_shot_y = [], []
                    
                    for cls in unique_classes:
                        cls_indices = np.where(y_train == cls)[0]
                        if len(cls_indices) >= k:
                            selected = rng.choice(cls_indices, k, replace=False)
                            few_shot_X.extend(X_train[selected])
                            few_shot_y.extend(y_train[selected])
                    
                    if len(few_shot_X) == 0:
                        continue
                    
                    few_shot_X = np.array(few_shot_X)
                    few_shot_y = np.array(few_shot_y)
                    
                    try:
                        # Clone the classifier for each trial
                        from sklearn.base import clone
                        clf_clone = clone(clf)
                        clf_clone.fit(few_shot_X, few_shot_y)
                        
                        # Test on subset for speed
                        test_size = min(len(X_test), 1000)
                        test_indices = rng.choice(len(X_test), test_size, replace=False)
                        
                        y_pred = clf_clone.predict(X_test[test_indices])
                        acc = accuracy_score(y_test[test_indices], y_pred)
                        accuracies.append(acc)
                    except Exception as e:
                        logger.warning(f"  {name} failed on trial {trial}: {e}")
                        continue
                
                if accuracies:
                    accs = np.array(accuracies)
                    results[k][name] = {
                        'mean': float(np.mean(accs)),
                        'std': float(np.std(accs)),
                        'ci_95': float(1.96 * np.std(accs) / np.sqrt(len(accs)))
                    }
                    logger.info(f"  {k}-shot {name}: {np.mean(accs):.4f}±{np.std(accs):.4f}")
        
        return results
    
    def evaluate_incremental(self, base_tasks, incremental_tasks,
                              class_names=None):
        """
        Evaluate class-incremental learning performance.
        Measures how well the model handles new classes without forgetting old ones.
        """
        logger.info("Evaluating class-incremental learning...")
        
        # Phase 1: Base class performance
        base_results = self.evaluate_few_shot(base_tasks)
        
        # Phase 2: Performance after adding novel classes
        incremental_results = self.evaluate_few_shot(incremental_tasks)
        
        # Calculate forgetting metric
        forgetting = base_results['accuracy']['mean'] - incremental_results['accuracy']['mean']
        
        results = {
            'base_accuracy': base_results['accuracy'],
            'incremental_accuracy': incremental_results['accuracy'],
            'forgetting': float(forgetting),
            'base_f1': base_results['macro_metrics']['f1_score'],
            'incremental_f1': incremental_results['macro_metrics']['f1_score']
        }
        
        logger.info(f"Base Acc: {base_results['accuracy']['mean']:.4f}")
        logger.info(f"Incremental Acc: {incremental_results['accuracy']['mean']:.4f}")
        logger.info(f"Forgetting: {forgetting:.4f}")
        
        return results
    
    # ======================== Visualization Methods ========================
    
    def plot_confusion_matrix(self, cm, class_names, title='Confusion Matrix',
                              filename='confusion_matrix.png'):
        """Generate publication-quality confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'},
            linewidths=0.5, linecolor='gray',
            square=True
        )
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {filepath}")
    
    def plot_tsne_embeddings(self, X, y, class_names=None, 
                              filename='tsne_embeddings.png',
                              n_samples=2000):
        """Generate t-SNE visualization of learned embeddings."""
        logger.info("Computing t-SNE embeddings...")
        
        # Sample if too many
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Process in batches to avoid OOM
            embeddings = []
            batch_size = 256
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                emb = self.model.embedding_net(batch)
                embeddings.append(emb.cpu().numpy())
            
            embeddings = np.concatenate(embeddings, axis=0)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(y)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            name = class_names[label] if class_names else f'Class {label}'
            plt.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=[colors[i]], label=name, alpha=0.6, s=20, edgecolors='none'
            )
        
        plt.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Learned Embeddings', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE plot to {filepath}")
    
    def plot_k_shot_comparison(self, meta_results, baseline_results,
                                filename='k_shot_comparison.png'):
        """Plot K-shot accuracy comparison: Meta-learning vs baselines."""
        plt.figure(figsize=(10, 6))
        
        k_shots = sorted(meta_results.keys())
        
        # Meta-learning (ours)
        meta_means = [meta_results[k]['mean'] for k in k_shots]
        meta_cis = [meta_results[k]['ci_95'] for k in k_shots]
        plt.errorbar(k_shots, meta_means, yerr=meta_cis, 
                     marker='o', linewidth=2, markersize=8,
                     label='MetaShield (Ours)', capsize=5, color='#00ff88')
        
        # Baselines
        baseline_colors = {'SVM': '#ff4444', 'Random Forest': '#ffaa00', 'KNN': '#00aaff'}
        
        for name in ['SVM', 'Random Forest', 'KNN']:
            means = []
            cis = []
            valid_k = []
            for k in k_shots:
                if k in baseline_results and name in baseline_results[k]:
                    means.append(baseline_results[k][name]['mean'])
                    cis.append(baseline_results[k][name]['ci_95'])
                    valid_k.append(k)
            
            if means:
                plt.errorbar(valid_k, means, yerr=cis, 
                            marker='s', linewidth=1.5, markersize=6,
                            label=name, capsize=5, 
                            color=baseline_colors.get(name, 'gray'),
                            linestyle='--')
        
        plt.xlabel('Number of Shots (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Few-Shot Classification Accuracy vs. Number of Shots', 
                   fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_shots)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved K-shot comparison to {filepath}")
    
    def plot_training_curves(self, history, filename='training_curves.png'):
        """Plot training loss/accuracy curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], label='Train', color='#00aaff', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], label='Validation', color='#ff4444', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], label='Train', color='#00aaff', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], label='Validation', color='#00ff88', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[2].plot(epochs, history['lr'], color='#ffaa00', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Learning Rate', fontsize=11)
        axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {filepath}")
    
    def plot_per_class_metrics(self, results, class_names,
                           filename='per_class_metrics.png'):
        """
        Plot per-class precision, recall, F1 bar chart.
        """

        precision = np.array(results['per_class']['precision'])
        recall = np.array(results['per_class']['recall'])
        f1 = np.array(results['per_class']['f1_score'])

        n_classes = min(
            len(class_names),
            len(precision),
            len(recall),
            len(f1)
        )

        class_names = class_names[:n_classes]
        precision = precision[:n_classes]
        recall = recall[:n_classes]
        f1 = f1[:n_classes]

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(n_classes)
        width = 0.25

        ax.bar(
            x - width,
            precision,
            width,
            label='Precision',
            color='#00aaff',
            alpha=0.85
        )

        ax.bar(
            x,
            recall,
            width,
            label='Recall',
            color='#00ff88',
            alpha=0.85
        )

        ax.bar(
            x + width,
            f1,
            width,
            label='F1-Score',
            color='#ffaa00',
            alpha=0.85
        )

        ax.set_xlabel('Attack Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(
            'Per-Class Classification Metrics',
            fontsize=14,
            fontweight='bold'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            class_names,
            rotation=30,
            ha='right'
        )

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        filepath = os.path.join(
            self.results_dir,
            'plots',
            filename
        )

        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        logger.info(
            f"Saved per-class metrics to {filepath}"
        )
    
    def generate_results_table(self, all_results, filename='results_table.json'):
        """Generate a formatted results table for the paper."""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Saved results table to {filepath}")
        return all_results
    
    def run_full_evaluation(self, X_train, y_train, X_test, y_test,
                             test_tasks, class_names,
                             training_history=None):
        """
        Run the complete evaluation pipeline.
        
        Generates all plots and metrics needed for the research paper.
        """
        logger.info("=" * 60)
        logger.info("RUNNING FULL EVALUATION PIPELINE")
        logger.info("=" * 60)
        
        all_results = {}
        
        # 1. Few-shot accuracy
        logger.info("\n[1/6] Few-shot classification evaluation...")
        few_shot_results = self.evaluate_few_shot(test_tasks, class_names)
        all_results['few_shot'] = few_shot_results
        
        # Plot confusion matrix
        if few_shot_results.get('confusion_matrix'):
            cm = np.array(few_shot_results['confusion_matrix'])
            n_classes = min(len(class_names), cm.shape[0])
            self.plot_confusion_matrix(cm, class_names[:n_classes])
        
        # 2. K-shot sweep
        logger.info("\n[2/6] K-shot sweep evaluation...")
        k_shot_results = self.evaluate_k_shot_sweep(
            X_test, y_test, k_shots=[1, 3, 5, 10], n_episodes=100
        )
        all_results['k_shot_sweep'] = k_shot_results
        
        # 3. Baseline comparison
        logger.info("\n[3/6] Baseline comparison...")
        baseline_results = self.evaluate_baselines(
            X_train, y_train, X_test, y_test, k_shots=[1, 5, 10]
        )
        all_results['baselines'] = baseline_results
        
        # Plot comparison
        self.plot_k_shot_comparison(k_shot_results, baseline_results)
        
        # 4. Per-class metrics
        logger.info("\n[4/6] Per-class metrics...")
        try:
            self.plot_per_class_metrics(
                few_shot_results,
                class_names
            )
        except Exception as e:
            logger.warning(
                f"Could not generate per-class metrics plot: {e}"
            )
        
        # 5. t-SNE visualization
        logger.info("\n[5/6] t-SNE visualization...")
        self.plot_tsne_embeddings(X_test, y_test, class_names)
        
        # 6. Training curves
        if training_history:
            logger.info("\n[6/6] Training curves...")
            self.plot_training_curves(training_history)
        
        # Save all results
        self.generate_results_table(all_results)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to {self.results_dir}/")
        logger.info("=" * 60)
        
        # Print summary for paper
        self._print_paper_summary(all_results)
        
        return all_results
    
    def _print_paper_summary(self, results):
        """Print a formatted summary suitable for the research paper."""
        logger.info("\n" + "=" * 60)
        logger.info("PAPER-READY RESULTS SUMMARY")
        logger.info("=" * 60)
        
        # Few-shot results
        fs = results.get('few_shot', {})
        acc = fs.get('accuracy', {})
        macro = fs.get('macro_metrics', {})
        
        logger.info(f"\nFew-Shot Classification Performance:")
        logger.info(f"  Accuracy: {acc.get('mean', 0):.4f} ± {acc.get('ci_95', 0):.4f} (95% CI)")
        logger.info(f"  Precision (macro): {macro.get('precision', 0):.4f}")
        logger.info(f"  Recall (macro): {macro.get('recall', 0):.4f}")
        logger.info(f"  F1-Score (macro): {macro.get('f1_score', 0):.4f}")
        
        # K-shot sweep
        ks = results.get('k_shot_sweep', {})
        if ks:
            logger.info(f"\nK-Shot Performance:")
            for k, v in sorted(ks.items()):
                logger.info(f"  {k}-shot: {v['mean']:.4f} ± {v['ci_95']:.4f}")
        
        # Baselines
        bl = results.get('baselines', {})
        if bl:
            logger.info(f"\nBaseline Comparison:")
            for k, methods in sorted(bl.items()):
                logger.info(f"  {k}-shot:")
                for name, v in methods.items():
                    logger.info(f"    {name}: {v['mean']:.4f} ± {v.get('ci_95', v.get('std', 0)):.4f}")
