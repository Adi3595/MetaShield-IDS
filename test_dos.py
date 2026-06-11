"""
MetaShield — Denial of Service (DoS) Test Script
=================================================
Simulates DoS-Slowloris and DoS-Hulk attack subtypes.

Prerequisites:
  1. Make sure python api.py is running in another terminal.
"""

import requests
import numpy as np

BASE_URL = "http://localhost:5000"

# ANSI Colors
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

def get_dimension():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def clear_existing_signatures():
    """Remove all pre-loaded signatures to prevent interference from other attack prototypes."""
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    existing = stats.get("known_attacks", [])
    for name in existing:
        requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
    if existing:
        print(f"  {Y}🧹 Cleared {len(existing)} pre-loaded signatures: {existing}{X}")

def generate_dos_features(dim, rng, subtype="Slowloris"):
    """
    Generate synthetic features that mimic DoS attack patterns.
    DoS traffic: single source overwhelming a target, either with
    slow drip connections (Slowloris) or massive flooding (Hulk).
    """
    flow = np.zeros(dim, dtype=np.float32)
    
    quarter = dim // 4
    
    if subtype == "Slowloris":
        # Slowloris: many open connections, tiny trickle of data, very long duration
        flow[:quarter] = rng.uniform(0.0, 2.0, quarter)                     # HTTP ports
        flow[quarter:quarter*2] = rng.uniform(5.0, 8.0, quarter)            # Very long duration
        flow[quarter*2:quarter*3] = rng.uniform(-6.0, -4.0, quarter)        # Tiny payload trickle
        flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)           # Many open connections
    else:
        # Hulk: massive HTTP GET floods, high bandwidth, short bursts
        flow[:quarter] = rng.uniform(0.5, 2.5, quarter)                     # HTTP ports
        flow[quarter:quarter*2] = rng.uniform(-3.0, -1.0, quarter)          # Short burst duration
        flow[quarter*2:quarter*3] = rng.uniform(4.0, 7.0, quarter)          # Massive payload volume
        flow[quarter*3:] = rng.uniform(5.0, 8.0, dim - quarter*3)           # Extreme request rate
    
    return flow.tolist()

def simulate_dos(name, subtype, seed):
    dim = get_dimension()
    rng = np.random.RandomState(seed)
    
    print(f"\n  {B}{C}[DoS Simulation] Threat: {name}{X}")
    print(f"  {'─'*50}")
    
    # 1. Learn the signature with structured features
    support_set = [generate_dos_features(dim, rng, subtype) for _ in range(5)]
    res_learn = requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": support_set}).json()
    print(f"  1. Learn: {G}{res_learn.get('message')}{X}")
    
    # 2. Test detection (IDS)
    query_flow = generate_dos_features(dim, rng, subtype)
    res_detect = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
    detected = res_detect.get('attack')
    conf = res_detect.get('confidence', 0)
    match = "✅" if detected == name else "⚠️"
    color = G if detected == name else R
    print(f"  2. Detect: {color}{match} '{detected}' (Confidence: {conf:.4f}){X}")
    
    # 3. Block it (IPS)
    res_block = requests.post(f"{BASE_URL}/api/block", json={"attack_name": name}).json()
    print(f"  3. IPS Firewall: {Y}🔒 Blocked → {res_block.get('currently_blocked')}{X}")
    
    # 4. Repeat Attack
    res_repeat = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
    print(f"  4. Repeat:  Blocked = {res_repeat.get('blocked')} | {res_repeat.get('message')}")
    
    # Clean up block (keep signature registered)
    requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})

if __name__ == "__main__":
    print(f"\n{B}{'='*60}")
    print(f"  MetaShield — DoS Attack Simulation")
    print(f"{'='*60}{X}")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=3)
        print(f"  {G}✅ API Server is ONLINE.{X}")
    except Exception:
        print(f"  {R}❌ API Server OFFLINE. Run: python api.py{X}")
        exit(1)
    
    clear_existing_signatures()
    simulate_dos("DoS-Slowloris", "Slowloris", seed=100)
    simulate_dos("DoS-Hulk", "Hulk", seed=101)
    
    print(f"\n{B}{'='*60}")
    print(f"  ✅ DoS Simulation Complete")
    print(f"{'='*60}{X}\n")
