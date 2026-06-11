# client.py
import requests
import numpy as np
import time

# API endpoint
BASE_URL = "http://localhost:5000"

def test_api():
    print("🔄 Testing MetaShield API")
    print("="*50)
    
    # 1. Check health
    response = requests.get(f"{BASE_URL}/health")
    print(f"\n1. Health check: {response.json()}")
    
    # 2. Get stats
    response = requests.get(f"{BASE_URL}/stats")
    print(f"\n2. Current stats: {response.json()}")
    
    # 3. Test detection on normal traffic
    normal_flow = np.random.randn(78).tolist()
    response = requests.post(
        f"{BASE_URL}/detect",
        json={"features": normal_flow}
    )
    print(f"\n3. Normal traffic detection: {response.json()}")
    
    # 4. Test detection on attack traffic
    # Simulate DoS-like pattern
    attack_flow = (np.random.randn(78) * 1.5 + 2).tolist()
    response = requests.post(
        f"{BASE_URL}/detect",
        json={"features": attack_flow}
    )
    print(f"\n4. Attack traffic detection: {response.json()}")
    
    # 5. Learn a new attack
    print(f"\n5. Teaching detector about new attack...")
    new_attack_examples = [
        (np.random.randn(78) * 2.0 + 3).tolist() for _ in range(3)
    ]
    response = requests.post(
        f"{BASE_URL}/learn",
        json={
            "attack_name": "ZeroDay_2024",
            "examples": new_attack_examples
        }
    )
    print(f"   Response: {response.json()}")
    
    # 6. Test detection of new attack
    new_attack_flow = (np.random.randn(78) * 2.0 + 3).tolist()
    response = requests.post(
        f"{BASE_URL}/detect",
        json={"features": new_attack_flow}
    )
    print(f"\n6. New attack detection: {response.json()}")
    
    # 7. Get updated stats
    response = requests.get(f"{BASE_URL}/stats")
    print(f"\n7. Updated stats: {response.json()}")

if __name__ == "__main__":
    # Wait for server to start
    print("Waiting for API server...")
    time.sleep(2)
    test_api()