"""
MetaShield — Stealth Attack Simulation
======================================
Simulates an attack that slowly mutates from normal traffic into a malicious pattern.
Demonstrates the sensitivity of the IDS and its ability to catch low-confidence threats.
"""

import requests
import numpy as np
import time

BASE_URL = "http://localhost:5000"

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

def get_dim():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def detect(flow):
    return requests.post(f"{BASE_URL}/api/detect", json={"features": flow}).json()

def learn(name, examples):
    return requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": examples}).json()

def block(name):
    return requests.post(f"{BASE_URL}/api/block", json={"attack_name": name}).json()

def clear_existing_signatures():
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    existing = stats.get("known_attacks", [])
    for name in existing:
        requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
        
def get_normal_flow(dim, seed):
    rng = np.random.RandomState(seed)
    flow = np.zeros(dim, dtype=np.float32)
    quarter = dim // 4
    flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)
    flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)
    flow[quarter*2:quarter*3] = rng.uniform(-1.0, 1.0, quarter)
    flow[quarter*3:] = rng.uniform(-1.0, 1.0, dim - quarter*3)
    return flow

def get_malicious_flow(dim, seed):
    rng = np.random.RandomState(seed)
    flow = np.zeros(dim, dtype=np.float32)
    quarter = dim // 4
    # Stealth-Worm: unusual port, short connections, self-replicating small payloads
    flow[:quarter] = rng.uniform(3.0, 5.0, quarter)
    flow[quarter:quarter*2] = rng.uniform(-3.0, -1.0, quarter)
    flow[quarter*2:quarter*3] = rng.uniform(1.0, 3.0, quarter)
    flow[quarter*3:] = rng.uniform(4.0, 6.0, dim - quarter*3)
    return flow

if __name__ == "__main__":
    print(f"\n{B}{C}=== MetaShield: Stealth Attack Simulation ==={X}\n")
    dim = get_dim()

    print(f"{C}[0] Clearing existing signatures...{X}")
    clear_existing_signatures()

    print(f"{C}[1] Defining malicious 'Stealth-Worm' pattern...{X}")
    malicious_base = get_malicious_flow(dim, 42)
    
    print(f"{C}[2] Teaching system the full 'Stealth-Worm' signature...{X}")
    # Train on the fully manifested attack
    support = [get_malicious_flow(dim, 42+i).tolist() for i in range(5)]
    learn("Stealth-Worm", support)
    
    print(f"\n{C}[3] Simulation: Normal traffic slowly mutating into Stealth-Worm...{X}")
    print(f"    (Watch the AI Confidence score rise as the attack manifests)\n")
    
    for step in range(11):
        alpha = step / 10.0  # 0.0 (normal) to 1.0 (full attack)
        
        # Mix normal noise with malicious signature
        normal_base = get_normal_flow(dim, 100 + step)
        flow = (1.0 - alpha) * normal_base + alpha * malicious_base
        
        res = detect(flow.tolist())
        
        is_threat = res.get("threat_detected")
        conf = res.get("confidence", 0.0)
        
        intensity = f"Mutation: {alpha*100:3.0f}%"
        if is_threat:
            if conf > 0.8:
                print(f"[{intensity}] {R}🚨 FULL ALERT  (Conf: {conf*100:4.1f}%){X}")
            else:
                print(f"[{intensity}] {Y}⚠️ EARLY WARN  (Conf: {conf*100:4.1f}%){X}")
        else:
            print(f"[{intensity}] {G}✅ CLEAN       (Conf: {conf*100:4.1f}%){X}")
            
        time.sleep(0.3)
        
    print(f"\n{C}[4] Auto-Blocking Stealth-Worm now that it's fully detected...{X}")
    block("Stealth-Worm")
    
    print(f"\n{C}[5] Follow-up flow (100% Mutation)...{X}")
    res = detect(malicious_base.tolist())
    if res.get("blocked"):
        print(f"[{'Mutation: 100%':14s}] {Y}🔒 BLOCKED     (Conf: {res.get('confidence',0)*100:4.1f}%){X}")
    
    print(f"\n{B}{G}=== Simulation Complete ==={X}\n")
