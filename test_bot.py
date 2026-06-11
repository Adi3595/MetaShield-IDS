"""
MetaShield — Botnet (Bot) Test Script
======================================
Simulates Botnet-Ares and Botnet-Zeus attack subtypes.

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

def generate_botnet_features(dim, rng, subtype="Ares"):
    """
    Generate synthetic features that mimic Botnet C2 traffic patterns.
    Botnet traffic: periodic beaconing, medium payload, encrypted C2 channels,
    consistent outbound connections to few destinations.
    """
    flow = np.zeros(dim, dtype=np.float32)
    
    quarter = dim // 4
    
    if subtype == "Ares":
        # Ares botnet: HTTP-based C2, periodic beacons, moderate packet sizes
        flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)                    # Normal port range
        flow[quarter:quarter*2] = rng.uniform(1.5, 4.0, quarter)            # Regular intervals (beaconing)
        flow[quarter*2:quarter*3] = rng.uniform(0.5, 2.5, quarter)          # Medium payload (commands)
        flow[quarter*3:] = rng.uniform(3.0, 6.0, dim - quarter*3)           # High outbound ratio
    else:
        # Zeus botnet: HTTPS-based C2, data exfiltration, larger payloads
        flow[:quarter] = rng.uniform(-0.5, 1.5, quarter)                    # HTTPS ports
        flow[quarter:quarter*2] = rng.uniform(2.0, 5.0, quarter)            # Long-lived connections
        flow[quarter*2:quarter*3] = rng.uniform(2.0, 5.0, quarter)          # Large payload (exfil)
        flow[quarter*3:] = rng.uniform(2.5, 5.5, dim - quarter*3)           # High outbound + encrypted
    
    return flow.tolist()

def simulate_bot(name, subtype, seed):
    dim = get_dimension()
    rng = np.random.RandomState(seed)
    
    print(f"\n  {B}{C}[Botnet Simulation] Threat: {name}{X}")
    print(f"  {'─'*50}")
    
    # 1. Learn the signature with structured features
    support_set = [generate_botnet_features(dim, rng, subtype) for _ in range(5)]
    res_learn = requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": support_set}).json()
    print(f"  1. Learn: {G}{res_learn.get('message')}{X}")
    
    # 2. Test detection (IDS)
    query_flow = generate_botnet_features(dim, rng, subtype)
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
    print(f"  MetaShield — Botnet Attack Simulation")
    print(f"{'='*60}{X}")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=3)
        print(f"  {G}✅ API Server is ONLINE.{X}")
    except Exception:
        print(f"  {R}❌ API Server OFFLINE. Run: python api.py{X}")
        exit(1)
    
    clear_existing_signatures()
    simulate_bot("Botnet-Ares", "Ares", seed=500)
    simulate_bot("Botnet-Zeus", "Zeus", seed=501)
    
    print(f"\n{B}{'='*60}")
    print(f"  ✅ Botnet Simulation Complete")
    print(f"{'='*60}{X}\n")
