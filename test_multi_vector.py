"""
MetaShield — Multi-Vector Attack Simulation
===========================================
Simulates a coordinated attack using multiple threat vectors (DDoS, SQLi, PortScan).
Demonstrates learning and blocking multiple distinct signatures simultaneously.
"""

import requests
import numpy as np
import time

BASE_URL = "http://localhost:5000"

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

def step(n, t): print(f"\n{B}{Y}[STEP {n}] {t}\n{'-'*60}{X}")

def clear_existing_signatures():
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    existing = stats.get("known_attacks", [])
    for name in existing:
        requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})

def attack_flows(name, n, dim, seed=0):
    rng = np.random.RandomState(42 + seed)
    flows = []
    quarter = dim // 4
    
    for _ in range(n):
        flow = np.zeros(dim, dtype=np.float32)
        if name == "SYN-Flood":
            flow[:quarter] = rng.uniform(4.0, 8.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(-6.0, -4.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(-4.0, -2.0, quarter)
            flow[quarter*3:] = rng.uniform(6.0, 10.0, dim - quarter*3)
        elif name == "SQLi-Wave":
            flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(2.0, 5.0, quarter)
            flow[quarter*3:] = rng.uniform(-3.0, -0.5, dim - quarter*3)
        elif name == "Ransomware":
            flow[:quarter] = rng.uniform(3.0, 6.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(4.0, 7.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(5.0, 8.0, quarter)
            flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)
        else:
            flow = rng.uniform(-1.0, 1.0, dim)
        flows.append(flow.tolist())
    
    return flows

def get_dim():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def detect(flow):
    return requests.post(f"{BASE_URL}/api/detect", json={"features": flow}).json()

def learn(name, examples):
    return requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": examples}).json()

def block(name):
    return requests.post(f"{BASE_URL}/api/block", json={"attack_name": name}).json()

def show_result(i, r, vector_name="Flow"):
    is_blocked = r.get("blocked", False)
    detected   = r.get("threat_detected", False)
    atk        = r.get("attack", "—")
    
    if is_blocked:
        print(f"  {vector_name} {i}: {Y}🔒 BLOCKED ({atk}){X}")
    elif detected:
        print(f"  {vector_name} {i}: {R}🚨 ALERT ({atk}){X}")
    else:
        print(f"  {vector_name} {i}: {G}✅ Clean{X}")

if __name__ == "__main__":
    print(f"\n{B}{C}=== MetaShield: Multi-Vector Attack Simulation ==={X}")
    try:
        dim = get_dim()
    except:
        print(f"{R}Server offline. Run python api.py first.{X}")
        exit(1)

    vectors = ["SYN-Flood", "SQLi-Wave", "Ransomware"]

    print(f"{C}Clearing existing signatures...{X}")
    clear_existing_signatures()

    step(1, "Phase 1: Coordinated Attack Begins (Unknown Threats)")
    for vec in vectors:
        print(f"\n  {C}--- Incoming Vector: {vec} ---{X}")
        flows = attack_flows(vec, 2, dim, seed=1)
        for i, f in enumerate(flows):
            show_result(i+1, detect(f), vec)
            
    step(2, "Phase 2: Rapid AI Antibody Synthesis (Learning)")
    for vec in vectors:
        print(f"  {C}Synthesizing prototype for {vec}...{X}", end=" ")
        res = learn(vec, attack_flows(vec, 5, dim, seed=2))
        if res.get("success"):
            print(f"{G}Done!{X}")
            
    step(3, "Phase 3: Attack Continues (IDS Alerts Active)")
    for vec in vectors:
        print(f"\n  {C}--- Incoming Vector: {vec} ---{X}")
        flows = attack_flows(vec, 2, dim, seed=3)
        for i, f in enumerate(flows):
            show_result(i+1, detect(f), vec)

    step(4, "Phase 4: Full IPS Lockdown (Blocking all vectors)")
    for vec in vectors:
        block(vec)
    print(f"  {G}Firewall rules updated. All known vectors blocked.{X}")

    step(5, "Phase 5: Attack Defeated (IPS Active)")
    for vec in vectors:
        print(f"\n  {C}--- Incoming Vector: {vec} ---{X}")
        flows = attack_flows(vec, 2, dim, seed=4)
        for i, f in enumerate(flows):
            show_result(i+1, detect(f), vec)
            
    print(f"\n{B}{G}=== Simulation Complete: System Secure ==={X}\n")
