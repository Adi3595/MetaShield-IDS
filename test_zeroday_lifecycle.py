"""
MetaShield — Zero-Day Attack Lifecycle Simulation
==================================================
This script demonstrates the dynamic zero-day threat lifecycle step-by-step:
1. Fire a Zero-Day Attack (Heartbleed) -> System is blind, passes through.
2. Teach the System (Few-Shot Vaccination) -> Learn from 5 examples.
3. Fire the Attack again -> Alert raised (IDS active).
4. Block the Attack -> Upgrade to IPS mode.
5. Repeat the Attack -> Instantly dropped by the firewall.
"""

import requests
import numpy as np
import time

BASE_URL = "http://localhost:5000"

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  M = "\033[95m"; X = "\033[0m"

def heading(text):
    print(f"\n{B}{M}{'='*75}\n  {text}\n{'='*75}{X}")

def step(n, text):
    print(f"\n{B}{Y}[STEP {n}] {text}\n{'-'*65}{X}")

def get_dimension():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def generate_threat_vector(dim, seed):
    """Generate features mimicking Heartbleed exploit traffic:
    TLS heartbeat requests with oversized response (memory dump)."""
    rng = np.random.RandomState(seed)
    flow = np.zeros(dim, dtype=np.float32)
    quarter = dim // 4
    
    # Heartbleed: TLS port, tiny request, massive response (leaked memory)
    flow[:quarter] = rng.uniform(1.0, 3.0, quarter)                  # TLS/HTTPS ports
    flow[quarter:quarter*2] = rng.uniform(-2.0, 0.0, quarter)        # Short duration
    flow[quarter*2:quarter*3] = rng.uniform(-5.0, -3.0, quarter)     # Tiny heartbeat request
    flow[quarter*3:] = rng.uniform(6.0, 9.0, dim - quarter*3)        # Massive leaked response
    
    return flow.tolist()

def main():
    heading("MetaShield — Zero-Day Threat Vaccination Lifecycle")
    
    # 0. Check Server
    try:
        dim = get_dimension()
    except Exception:
        print(f"  {R}❌ Server offline. Run python api.py first.{X}")
        return

    # Clean up previous states to ensure a fresh demo run
    requests.post(f"{BASE_URL}/api/forget", json={"attack_name": "Heartbleed-ZeroDay"})
    requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": "Heartbleed-ZeroDay"})
    
    # --- STEP 1: Blind Attack ---
    step(1, "Zero-Day Attack Injected (Before Learning)")
    print(f"  {C}Target threat: 'Heartbleed-ZeroDay'{X}")
    print(f"  {C}Sending 3 exploit network packets...{X}")
    
    # Generate mock heartbleed traffic
    exploit_packets = [generate_threat_vector(dim, seed=i) for i in range(3)]
    
    for i, packet in enumerate(exploit_packets):
        res = requests.post(f"{BASE_URL}/api/detect", json={"features": packet}).json()
        threat_detected = res.get("threat_detected")
        pred_attack = res.get("attack")
        confidence = res.get("confidence", 0)
        
        if threat_detected and pred_attack == "Heartbleed-ZeroDay":
            print(f"  Packet {i+1}: {R}🚨 Detected as {pred_attack} (conf: {confidence:.2f}){X}")
        else:
            # Expected behavior: The system either flags it as clean or misclassifies it because it has no prototype
            misclass = f" (misclassified as '{pred_attack}')" if pred_attack else " (classified as Clean)"
            print(f"  Packet {i+1}: {G}✅ Exploit traffic passed through successfully{misclass}{X}")

    # --- STEP 2: Few-Shot Vaccination ---
    step(2, "Synthesizing Antibody (Few-Shot learning)")
    print(f"  {C}Collecting 5 exploit payloads captured in sandbox...{X}")
    support_set = [generate_threat_vector(dim, seed=i+10) for i in range(5)]
    
    print(f"  {C}Submitting 5 samples to `/api/learn` to construct Bayesian prototype...{X}")
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/api/learn", json={
        "attack_name": "Heartbleed-ZeroDay",
        "examples": support_set
    }).json()
    dt = (time.time() - t0) * 1000
    
    if res.get("success"):
        print(f"  {G}✅ Antibody created successfully in {dt:.1f}ms!{X}")
    else:
        print(f"  {R}❌ Learning failed.{X}")
        return

    # --- STEP 3: IDS Alert Detection ---
    step(3, "IDS Mode Activation (Scanning Same Threat after Learning)")
    print(f"  {C}Sending new, slightly varied exploit packets...{X}")
    new_exploits = [generate_threat_vector(dim, seed=i+20) for i in range(3)]
    
    for i, packet in enumerate(new_exploits):
        res = requests.post(f"{BASE_URL}/api/detect", json={"features": packet}).json()
        threat_detected = res.get("threat_detected")
        pred_attack = res.get("attack")
        confidence = res.get("confidence", 0)
        
        if threat_detected and pred_attack == "Heartbleed-ZeroDay":
            print(f"  Packet {i+1}: {R}🚨 ALERT! Identified threat '{pred_attack}' (confidence: {confidence*100:.1f}%){X}")
            print(f"           ↳ Message: {res.get('message')}")
        else:
            print(f"  Packet {i+1}: {G}✅ Missed threat (conf: {confidence*100:.1f}%){X}")

    # --- STEP 4: Upgrade to IPS Firewall Block ---
    step(4, "Upgrading Security Shield to IPS (Block List)")
    print(f"  {C}Enabling firewall block rule for 'Heartbleed-ZeroDay'...{X}")
    res = requests.post(f"{BASE_URL}/api/block", json={"attack_name": "Heartbleed-ZeroDay"}).json()
    if res.get("success"):
        print(f"  {G}🔒 Firewall rule engaged! Blocked categories: {res.get('currently_blocked')}{X}")

    # --- STEP 5: Drop Intercepted Packets ---
    step(5, "Attack Blocked (IPS Active)")
    print(f"  {C}Attacker fires Heartbleed exploit again...{X}")
    repeat_exploits = [generate_threat_vector(dim, seed=i+30) for i in range(3)]
    
    for i, packet in enumerate(repeat_exploits):
        res = requests.post(f"{BASE_URL}/api/detect", json={"features": packet}).json()
        blocked = res.get("blocked", False)
        pred_attack = res.get("attack")
        
        if blocked:
            print(f"  Packet {i+1}: {Y}🔒 INTERCEPTED & DROPPED! Rule matched: '{pred_attack}'{X}")
            print(f"           ↳ Firewall message: {res.get('message')}")
        else:
            print(f"  Packet {i+1}: {R}🚨 Leak! Packet was allowed through.{X}")

    # --- STEP 6: Unblock & Revert ---
    step(6, "Post-Incident Audit (Reverting to passive IDS mode)")
    requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": "Heartbleed-ZeroDay"})
    print(f"  {G}Removed firewall rule. Threat log is secure.{X}")
    
    heading("Zero-Day Vaccine Lifecycle Complete")

if __name__ == "__main__":
    main()
