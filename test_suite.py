# test_suite.py
import torch
import numpy as np
from src.models.prototypical_network import PrototypicalNetwork
import time
import json

class ModelTestSuite:
    def __init__(self, model_path='checkpoints/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PrototypicalNetwork(input_dim=78, embedding_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.results = {}
        
    def test_adaptation_speed(self, n_tests=100):
        """Test how fast model adapts to new attacks"""
        print("\n⏱️  Testing Adaptation Speed...")
        
        times = []
        for _ in range(n_tests):
            # Create random support set (new attack)
            support_x = torch.FloatTensor(np.random.randn(5, 78)).to(self.device)
            support_y = torch.LongTensor([0]*5).to(self.device)
            
            # Time the adaptation
            start = time.time()
            with torch.no_grad():
                # FIXED: Changed from self.model.encoder to self.model.embedding_net
                embeddings = self.model.embedding_net(support_x)
                prototype = embeddings.mean(dim=0)
            end = time.time()
            
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        self.results['adaptation_speed'] = {
            'average_ms': avg_time,
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'samples': n_tests
        }
        
        print(f"   Average adaptation time: {avg_time:.2f} ms")
        print(f"   Fastest: {np.min(times):.2f} ms")
        print(f"   Slowest: {np.max(times):.2f} ms")
        
    def test_few_shot_accuracy(self, n_way=5, k_shot=1, n_tests=50):
        """Test accuracy with different shot counts"""
        print(f"\n🎯 Testing {k_shot}-shot Learning Accuracy...")
        
        accuracies = []
        
        for test in range(n_tests):
            # Create synthetic classes
            class_centers = np.random.randn(n_way, 78) * 2
            
            # Create support set
            support_x = []
            support_y = []
            for i, center in enumerate(class_centers):
                for _ in range(k_shot):
                    example = center + np.random.randn(78) * 0.3
                    support_x.append(example)
                    support_y.append(i)
            
            # Create query set
            query_x = []
            query_y = []
            for i, center in enumerate(class_centers):
                for _ in range(10):  # 10 query samples per class
                    example = center + np.random.randn(78) * 0.5
                    query_x.append(example)
                    query_y.append(i)
            
            # Convert to tensors
            support_x = torch.FloatTensor(support_x).to(self.device)
            support_y = torch.LongTensor(support_y).to(self.device)
            query_x = torch.FloatTensor(query_x).to(self.device)
            query_y = torch.LongTensor(query_y).to(self.device)
            
            # Test
            with torch.no_grad():
                logits = self.model(support_x, support_y, query_x)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_y).float().mean().item()
                accuracies.append(acc)
        
        avg_acc = np.mean(accuracies)
        self.results[f'{k_shot}_shot_accuracy'] = {
            'average': avg_acc,
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        }
        
        print(f"   Average accuracy: {avg_acc:.2%}")
        print(f"   Std deviation: {np.std(accuracies):.2%}")
        
    def test_robustness(self, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """Test robustness to noise"""
        print("\n🔧 Testing Robustness to Noise...")
        
        base_center = np.random.randn(78)
        
        for noise in noise_levels:
            # Create support set (clean)
            support_x = []
            for _ in range(5):
                example = base_center + np.random.randn(78) * 0.1
                support_x.append(example)
            
            # Create query set with noise
            query_x = []
            for _ in range(50):
                example = base_center + np.random.randn(78) * noise
                query_x.append(example)
            
            # Convert to tensors
            support_x = torch.FloatTensor(support_x).to(self.device)
            support_y = torch.LongTensor([0]*5).to(self.device)
            query_x = torch.FloatTensor(query_x).to(self.device)
            query_y = torch.LongTensor([0]*50).to(self.device)
            
            # Test
            with torch.no_grad():
                logits = self.model(support_x, support_y, query_x)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_y).float().mean().item()
            
            self.results[f'noise_{noise}'] = acc
            print(f"   Noise {noise:.0%}: Accuracy {acc:.2%}")
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("="*60)
        print("🔬 META-SHIELD COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        self.test_adaptation_speed()
        self.test_few_shot_accuracy(k_shot=1)
        self.test_few_shot_accuracy(k_shot=3)
        self.test_few_shot_accuracy(k_shot=5)
        self.test_robustness()
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ TESTS COMPLETE")
        print("="*60)
        print(f"\nResults saved to test_results.json")
        
        # Summary
        print("\n📊 SUMMARY:")
        print(f"   Adaptation Speed: {self.results['adaptation_speed']['average_ms']:.1f} ms")
        print(f"   1-shot Accuracy: {self.results['1_shot_accuracy']['average']:.2%}")
        print(f"   3-shot Accuracy: {self.results['3_shot_accuracy']['average']:.2%}")
        print(f"   5-shot Accuracy: {self.results['5_shot_accuracy']['average']:.2%}")
        
        # Overall assessment
        overall_score = (
            self.results['1_shot_accuracy']['average'] * 0.4 +
            self.results['3_shot_accuracy']['average'] * 0.3 +
            self.results['5_shot_accuracy']['average'] * 0.3
        )
        
        print(f"\n🏆 Overall Score: {overall_score:.2%}")
        
        if overall_score > 0.9:
            print("   Status: EXCELLENT - Ready for production")
        elif overall_score > 0.8:
            print("   Status: GOOD - Ready for testing")
        elif overall_score > 0.7:
            print("   Status: ACCEPTABLE - Needs more training")
        else:
            print("   Status: NEEDS IMPROVEMENT - Retrain with more epochs")

if __name__ == "__main__":
    tester = ModelTestSuite()
    tester.run_comprehensive_test()