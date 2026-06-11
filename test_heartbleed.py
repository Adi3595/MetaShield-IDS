"""
MetaShield — Heartbleed Test Script
====================================
Simulates Heartbleed-CVE-2014-0160 attack subtype.

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
    """Remove all pre-loaded signatures to prevent interference from demo prototypes."""
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    existing = stats.get("known_attacks", [])
    for name in existing:
        requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
    if existing:
        print(f"  {Y}🧹 Cleared {len(existing)} pre-loaded signatures: {existing}{X}")

def simulate_heartbleed(name, bias, variance, seed):
    dim = get_dimension()
    rng = np.random.RandomState(seed)
    
    print(f"\n  {B}{C}[Heartbleed Simulation] Threat: {name}{X}")
    print(f"  {'─'*50}")
    
    # 1. Learn the signature
    support_set = [(rng.randn(dim) * variance + bias).tolist() for _ in range(5)]
    res_learn = requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": support_set}).json()
    print(f"  1. Learn: {G}{res_learn.get('message')}{X}")
    
    # 2. Test detection (IDS)
    query_flow = (rng.randn(dim) * variance + bias).tolist()
    res_detect = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
    detected = res_detect.get('attack')
    conf = res_detect.get('confidence', 0)
    match = "✅" if detected == name else "⚠️"
    color = G if detected == name else R
    print(f"  2. Detect: {color}{match} '{detected}' (Confidence: {conf:.2f}){X}")
    
    # 3. Block it (IPS)
    res_block = requests.post(f"{BASE_URL}/api/block", json={"attack_name": name}).json()
    print(f"  3. IPS Firewall: {Y}🔒 Blocked → {res_block.get('currently_blocked')}{X}")
    
    # 4. Repeat Attack
    res_repeat = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
    print(f"  4. Repeat:  Blocked = {res_repeat.get('blocked')} | {res_repeat.get('message')}")
    
    # Clean up
    requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})

if __name__ == "__main__":
    print(f"\n{B}{'='*60}")
    print(f"  MetaShield — Heartbleed Attack Simulation")
    print(f"{'='*60}{X}")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=3)
        print(f"  {G}✅ API Server is ONLINE.{X}")
    except Exception:
        print(f"  {R}❌ API Server OFFLINE. Run: python api.py{X}")
        exit(1)
    
    # clear_existing_signatures() # Commented out so signatures accumulate in the registry and dashboard
    simulate_heartbleed("Heartbleed-CVE-2014-0160", 3.8, 0.5, seed=800)
    
    print(f"\n{B}{'='*60}")
    print(f"  ✅ Heartbleed Simulation Complete")
    print(f"{'='*60}{X}\n")
