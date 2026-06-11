"""
MetaShield — PortScan Test Script
=================================
Simulates PortScan-TCP-SYN and PortScan-Stealth attack subtypes.

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

def generate_portscan_features(dim, rng, subtype="TCP-SYN"):
    """
    Generate synthetic features that mimic PortScan traffic patterns.
    Uses structured feature patterns so the neural network produces
    distinguishable embeddings for each attack type.
    """
    flow = np.zeros(dim, dtype=np.float32)
    
    # PortScan signature: high activity in port-related features (first quarter),
    # very low payload, short duration, many connection attempts
    port_features = dim // 4
    
    if subtype == "TCP-SYN":
        # TCP SYN scan: rapid short connections, SYN flags, no payload
        flow[:port_features] = rng.uniform(3.0, 6.0, port_features)   # High port activity
        flow[port_features:port_features*2] = rng.uniform(-4.0, -2.0, port_features)  # Low duration
        flow[port_features*2:port_features*3] = rng.uniform(-3.0, -1.5, port_features) # Low payload
        flow[port_features*3:] = rng.uniform(0.5, 2.0, dim - port_features*3)  # Moderate flags
    else:
        # Stealth scan: slower, randomized ports, FIN/NULL/XMAS flags
        flow[:port_features] = rng.uniform(2.0, 4.5, port_features)   # Moderate port activity
        flow[port_features:port_features*2] = rng.uniform(-3.0, -1.0, port_features)  # Short duration
        flow[port_features*2:port_features*3] = rng.uniform(-5.0, -3.0, port_features) # Very low payload
        flow[port_features*3:] = rng.uniform(-2.0, 0.5, dim - port_features*3)  # Unusual flags
    
    return flow.tolist()

def simulate_portscan(name, subtype, seed):
    dim = get_dimension()
    rng = np.random.RandomState(seed)
    
    print(f"\n  {B}{C}[PortScan Simulation] Threat: {name}{X}")
    print(f"  {'─'*50}")
    
    # 1. Learn the signature with structured features
    support_set = [generate_portscan_features(dim, rng, subtype) for _ in range(5)]
    res_learn = requests.post(f"{BASE_URL}/api/learn", json={"attack_name": name, "examples": support_set}).json()
    print(f"  1. Learn: {G}{res_learn.get('message')}{X}")
    
    # 2. Test detection (IDS) — generate a new query from the same distribution
    query_flow = generate_portscan_features(dim, rng, subtype)
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
    print(f"  MetaShield — PortScan Attack Simulation")
    print(f"{'='*60}{X}")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=3)
        print(f"  {G}✅ API Server is ONLINE.{X}")
    except Exception:
        print(f"  {R}❌ API Server OFFLINE. Run: python api.py{X}")
        exit(1)
    
    clear_existing_signatures()
    simulate_portscan("PortScan-TCP-SYN", "TCP-SYN", seed=300)
    simulate_portscan("PortScan-Stealth", "Stealth", seed=301)
    
    print(f"\n{B}{'='*60}")
    print(f"  ✅ PortScan Simulation Complete")
    print(f"{'='*60}{X}\n")
