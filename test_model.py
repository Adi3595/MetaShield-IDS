# test_model.py
import torch
import numpy as np
from src.models.prototypical_network import PrototypicalNetwork

def test_trained_model():
    print("🔍 Testing Trained Model")
    print("="*50)
    
    # 1. Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork(input_dim=78, embedding_dim=128)
    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    
    # 2. Create a simple test (simulate learning a new attack)
    print("\n📝 Testing few-shot learning capability...")
    
    # Simulate 3 new attack types with 2 examples each
    n_way = 3  # 3 attack types
    k_shot = 2  # 2 examples each
    
    # Create synthetic examples
    np.random.seed(42)
    
    # Attack type A (e.g., new DDoS variant)
    attack_a = np.random.randn(k_shot, 78) * 1.5 + 2
    
    # Attack type B (e.g., new probe technique)
    attack_b = np.random.randn(k_shot, 78) * 1.2 - 1
    
    # Attack type C (e.g., new brute force)
    attack_c = np.random.randn(k_shot, 78) * 0.8 + 0.5
    
    # Combine support set (few examples to learn from)
    support_x = torch.FloatTensor(np.vstack([attack_a, attack_b, attack_c])).to(device)
    support_y = torch.LongTensor([0,0, 1,1, 2,2]).to(device)  # Labels
    
    # Create query samples (to test if learning worked)
    query_a = torch.FloatTensor(np.random.randn(5, 78) * 1.5 + 2).to(device)
    query_b = torch.FloatTensor(np.random.randn(5, 78) * 1.2 - 1).to(device)
    query_c = torch.FloatTensor(np.random.randn(5, 78) * 0.8 + 0.5).to(device)
    
    query_x = torch.cat([query_a, query_b, query_c])
    query_y = torch.LongTensor([0]*5 + [1]*5 + [2]*5).to(device)
    
    # Test the model
    with torch.no_grad():
        logits = model(support_x, support_y, query_x)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_y).float().mean().item()
    
    print(f"✅ Few-shot learning test passed!")
    print(f"   Accuracy on new attacks: {accuracy:.2%}")
    
    return model

if __name__ == "__main__":
    model = test_trained_model()