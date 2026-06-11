"""
MetaShield — IPS Stress Test
============================
Simulates a high-volume mix of normal traffic, known attacks (blocked),
and unknown attacks (pass through) to evaluate IPS throughput and filtering accuracy.
"""

import requests
import numpy as np
import time
import random

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

def generate_flow(type_, dim):
    rng = np.random.RandomState()
    flow = np.zeros(dim, dtype=np.float32)
    quarter = dim // 4
    
    if type_ == "normal":
        flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter*2:quarter*3] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter*3:] = rng.uniform(-1.0, 1.0, dim - quarter*3)
    elif type_ == "APT-X":
        # Slow exfiltration over specific ports
        flow[:quarter] = rng.uniform(2.0, 4.0, quarter)
        flow[quarter:quarter*2] = rng.uniform(5.0, 8.0, quarter)
        flow[quarter*2:quarter*3] = rng.uniform(3.0, 5.0, quarter)
        flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)
    elif type_ == "ZeroDay-Y":
        # Rapid exploits
        flow[:quarter] = rng.uniform(4.0, 7.0, quarter)
        flow[quarter:quarter*2] = rng.uniform(-5.0, -3.0, quarter)
        flow[quarter*2:quarter*3] = rng.uniform(-4.0, -2.0, quarter)
        flow[quarter*3:] = rng.uniform(6.0, 9.0, dim - quarter*3)
        
    return flow.tolist()

if __name__ == "__main__":
    print(f"\n{B}{C}=== MetaShield: IPS Stress Test ==={X}\n")
    dim = get_dim()

    print(f"{C}[0] Clearing existing signatures...{X}")
    clear_existing_signatures()

    print(f"{C}[1] Teaching system about 'APT-X'...{X}")
    learn("APT-X", [generate_flow("APT-X", dim) for _ in range(5)])
    
    print(f"{C}[2] Engaging IPS block for 'APT-X'...{X}")
    block("APT-X")

    print(f"\n{C}[3] Initiating Stress Test (100 flows mixed)...{X}")
    
    stats = {"normal_passed": 0, "apt_blocked": 0, "zeroday_passed": 0}
    t0 = time.time()
    
    for i in range(100):
        # 70% normal, 20% known attack (APT-X), 10% zero-day
        r = random.random()
        if r < 0.7:
            flow_type = "normal"
        elif r < 0.9:
            flow_type = "APT-X"
        else:
            flow_type = "ZeroDay-Y"
            
        flow = generate_flow(flow_type, dim)
        res = detect(flow)
        
        if res.get("blocked"):
            stats["apt_blocked"] += 1
            print(f"[{i:03d}] {Y}🔒 BLOCKED  (APT-X){X}")
        elif res.get("threat_detected"):
            stats["zeroday_passed"] += 1
            print(f"[{i:03d}] {R}🚨 ALERT    (ZeroDay-Y?){X}")
        else:
            stats["normal_passed"] += 1
            print(f"[{i:03d}] {G}✅ CLEAN    (Normal){X}")
            
    t1 = time.time()
    
    print(f"\n{B}{C}=== Stress Test Results ==={X}")
    print(f"  Time taken     : {t1-t0:.2f} seconds")
    print(f"  Normal allowed : {stats['normal_passed']}")
    print(f"  APT-X blocked  : {stats['apt_blocked']} (100% expected if IPS working)")
    print(f"  Unknown alerts : {stats['zeroday_passed']} (ZeroDays pass IPS filter but trigger IDS)")
    print("\n")
