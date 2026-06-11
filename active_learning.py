# active_learning.py
import numpy as np
from sklearn.cluster import KMeans

class ActiveLearner:
    """Actively selects which samples need labeling"""
    
    def __init__(self, detector, uncertainty_threshold=0.3):
        self.detector = detector
        self.threshold = uncertainty_threshold
        self.unlabeled_pool = []
        
    def uncertainty_sampling(self, samples, n_queries=5):
        """Select most uncertain samples for labeling"""
        uncertainties = []
        
        for sample in samples:
            _, confidence = self.detector.detect(sample)
            uncertainty = 1 - confidence
            uncertainties.append(uncertainty)
        
        # Get indices of most uncertain samples
        indices = np.argsort(uncertainties)[-n_queries:]
        return [samples[i] for i in indices]
    
    def diversity_sampling(self, samples, n_queries=5):
        """Select diverse samples using clustering"""
        if len(samples) < n_queries:
            return samples
        
        kmeans = KMeans(n_clusters=n_queries)
        clusters = kmeans.fit_predict(samples)
        
        # Select one sample from each cluster
        selected = []
        for i in range(n_queries):
            cluster_samples = samples[clusters == i]
            selected.append(cluster_samples[0])
        
        return selected
    
    def query_expert(self, samples):
        """Request labels from security analyst"""
        print("\n🔍 Active Learning Query")
        print("="*40)
        
        for i, sample in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Features: {sample[:5]}...")  # Show first 5 features
            
            # In real system, this would show actual traffic details
            label = input("  Enter attack type (or 'benign'): ")
            
            if label.lower() != 'benign':
                self.detector.learn_new_attack(label, [sample])
                print(f"  ✅ Learned new attack: {label}")