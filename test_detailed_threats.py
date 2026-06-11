"""
MetaShield — Detailed Threat Subtype Simulation
===============================================
This script registers and tests 8 distinct attack families and their subtypes.
It sends support samples to teach the network, query samples to verify IDS alarms,
and triggers the IPS blocking rules.

Prerequisites:
  1. Make sure python api.py is running in another terminal.
  2. For best accuracy, run python main.py --quick first to train the network model.
"""

import requests
import numpy as np
import time

BASE_URL = "http://localhost:5000"

# ANSI Colors
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  M = "\033[95m"; X = "\033[0m"

# Threat Taxonomy with Subtypes
THREAT_TAXONOMY = {
    "DoS-Slowloris": {
        "family": "DoS",
        "subtype": "Slowloris (Slow HTTP Denial of Service)",
        "description": "Establishes multiple connections to the target web server and keeps them open as long as possible."
    },
    "DoS-Hulk": {
        "family": "DoS",
        "subtype": "Hulk (HTTP Unbearable Load King)",
        "description": "Generates massive volumes of unique HTTP requests to bypass caching and exhaust web server threads."
    },
    "DDoS-SYN-Flood": {
        "family": "DDoS",
        "subtype": "SYN-Flood (TCP Synchronize Attack)",
        "description": "Floods a target system with TCP SYN packets to exhaust connection queues."
    },
    "WebAttack-SQLi": {
        "family": "WebAttack",
        "subtype": "SQL Injection (SQLi Injection)",
        "description": "Injects malicious SQL queries into user input fields to read/manipulate database tables."
    },
    "WebAttack-XSS": {
        "family": "WebAttack",
        "subtype": "Cross-Site Scripting (XSS)",
        "description": "Injects client-side scripts into trusted web pages to hijack user sessions."
    },
    "BruteForce-SSH": {
        "family": "BruteForce",
        "subtype": "SSH-Patator (Credential Stuffing)",
        "description": "Attempts to gain unauthorized access to an SSH port via rapid dictionary credential testing."
    },
    "Botnet-Ares": {
        "family": "Bot",
        "subtype": "Ares Command & Control (C2)",
        "description": "Exchanges command packets and system telemetry between compromised bots and a central C2 server."
    },
    "Infiltration-Exploit": {
        "family": "Infiltration",
        "subtype": "Privilege Escalation Exploit",
        "description": "Exploits system vulnerabilities to elevate access privileges inside a restricted network node."
    }
}

def heading(text):
    print(f"\n{B}{M}{'='*75}\n  {text}\n{'='*75}{X}")

def step(n, text):
    print(f"\n{B}{Y}[STEP {n}] {text}\n{'-'*65}{X}")

def generate_mock_flow(profile, dim, seed):
    rng = np.random.RandomState(seed)
    flow = np.zeros(dim, dtype=np.float32)
    quarter = dim // 4
    family = profile["family"]
    
    if family == "DoS":
        flow[:quarter] = rng.uniform(0.0, 2.0, quarter)                     # HTTP ports
        flow[quarter:quarter*2] = rng.uniform(4.0, 8.0, quarter)            # Extreme values
        flow[quarter*2:quarter*3] = rng.uniform(-6.0, -4.0, quarter)        # Payload shift
        flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)           # Connection counts
    elif family == "DDoS":
        flow[:quarter] = rng.uniform(4.0, 8.0, quarter)                     # Spoofed IPs
        flow[quarter:quarter*2] = rng.uniform(-6.0, -4.0, quarter)          # No connection duration
        flow[quarter*2:quarter*3] = rng.uniform(-4.0, -2.0, quarter)        # Tiny packets
        flow[quarter*3:] = rng.uniform(6.0, 10.0, dim - quarter*3)          # Extreme packet rate
    elif family == "WebAttack":
        flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)                    # HTTP ports
        flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)           # Normal duration
        flow[quarter*2:quarter*3] = rng.uniform(2.0, 5.0, quarter)          # Large request payloads
        flow[quarter*3:] = rng.uniform(-3.0, -0.5, dim - quarter*3)         # Varied responses
    elif family == "BruteForce":
        flow[:quarter] = rng.uniform(-3.0, -1.0, quarter)                   # Focused port range (SSH/FTP)
        flow[quarter:quarter*2] = rng.uniform(-5.0, -3.0, quarter)          # Very short connections
        flow[quarter*2:quarter*3] = rng.uniform(-2.0, 0.0, quarter)         # Small auth packets
        flow[quarter*3:] = rng.uniform(4.0, 7.0, dim - quarter*3)           # Very high connection count
    elif family == "Bot":
        flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)                    # Normal port range
        flow[quarter:quarter*2] = rng.uniform(1.5, 4.0, quarter)            # Beaconing intervals
        flow[quarter*2:quarter*3] = rng.uniform(0.5, 2.5, quarter)          # Medium payload
        flow[quarter*3:] = rng.uniform(3.0, 6.0, dim - quarter*3)           # High outbound ratio
    elif family == "Infiltration":
        flow[:quarter] = rng.uniform(-0.5, 1.5, quarter)                    # Mixed ports
        flow[quarter:quarter*2] = rng.uniform(-1.0, 2.0, quarter)           # Normal duration
        flow[quarter*2:quarter*3] = rng.uniform(-2.0, -1.0, quarter)        # Tiny request (exploit drop)
        flow[quarter*3:] = rng.uniform(5.0, 8.0, dim - quarter*3)           # Large reverse shell payload
    else:
        flow = rng.randn(dim) * 0.5 + 2.0
        
    return flow.tolist()

def check_server():
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=3).json()
        print(f"  {G}✅ API Server is ONLINE.{X} Model loaded: {res.get('model_loaded')}")
        return True
    except Exception:
        print(f"  {R}❌ API Server OFFLINE. Please run: python api.py in another terminal.{X}")
        return False

def get_dimension():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def clear_existing_signatures():
    """Remove all pre-existing signatures so they don't interfere with detection."""
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    existing = stats.get("known_attacks", [])
    for name in existing:
        requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
    if existing:
        print(f"  {Y}🧹 Cleared {len(existing)} pre-existing signatures: {existing}{X}")
    else:
        print(f"  {G}Registry is clean — no pre-existing signatures.{X}")

def main():
    heading("MetaShield — Advanced Subtype Threat Simulation")
    
    if not check_server():
        return
        
    dim = get_dimension()
    print(f"  Vector Input Dimension: {dim} features.")
    
    # ── Step 0: Clear any pre-existing prototypes ──
    step(0, "Clearing Pre-Existing Signatures")
    clear_existing_signatures()
    
    # ── Step 1: Pre-check status ──
    step(1, "Reviewing Current Registered Threat Signatures")
    stats = requests.get(f"{BASE_URL}/api/stats").json()
    print(f"  Existing signatures in registry: {stats['known_attacks']}")

    # ── Step 2: Learn and catalog all subtypes ──
    step(2, "Teaching the Prototypical Network 8 New Threat Subtypes")
    for name, info in THREAT_TAXONOMY.items():
        print(f"\n  📝 Attack: {B}{C}{name}{X}")
        print(f"     Subtype: {info['subtype']}")
        print(f"     Details: {info['description']}")
        
        # Generate 5 support examples (Few-shot learning)
        support_examples = [generate_mock_flow(info, dim, seed=i) for i in range(5)]
        
        # Inject to model prototype registry
        t0 = time.time()
        res = requests.post(f"{BASE_URL}/api/learn", json={
            "attack_name": name,
            "examples": support_examples
        }).json()
        dt_ms = (time.time() - t0) * 1000
        
        if res.get("success"):
            print(f"     {G}➔ AI Prototype Synthesized in {dt_ms:.1f}ms (5 shots used){X}")
        else:
            print(f"     {R}➔ Synthesis Failed.{X}")

    # ── Step 3: Test detection and IDS alarms ──
    step(3, "Testing Real-Time Threat Classification (IDS Mode)")
    for name, info in THREAT_TAXONOMY.items():
        print(f"\n  📡 Testing query flow for: {B}{C}{name}{X} ({info['subtype']})")
        
        # Generate 1 query flow (different seed to ensure similarity, not identity)
        query_flow = generate_mock_flow(info, dim, seed=100)
        
        # Query detection
        res = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
        
        threat_detected = res.get("threat_detected")
        pred_attack = res.get("attack")
        confidence = res.get("confidence", 0)
        
        if threat_detected:
            # First occurrence triggers auto-IPS activation
            print(f"     {R}🚨 ALERT! Threat: '{pred_attack}' (Certainty: {confidence*100:.1f}%) Severity: {res.get('severity')}{X}")
            print(f"     {Y}↳ Message: {res.get('message')}{X}")
        else:
            print(f"     {G}✅ Clean Flow (Certainty: {confidence*100:.1f}%){X}")

    # ── Step 4: Verify IPS blocks ──
    step(4, "Testing IPS Blocking & Firewall Interception")
    # Query another flow for a registered threat (which should now be actively blocked)
    target_threat = "WebAttack-SQLi"
    target_info = THREAT_TAXONOMY[target_threat]
    
    print(f"  Sending traffic matching previously seen: {B}{C}{target_threat}{X}")
    query_flow = generate_mock_flow(target_info, dim, seed=105)
    
    res = requests.post(f"{BASE_URL}/api/detect", json={"features": query_flow}).json()
    is_blocked = res.get("blocked", False)
    
    if is_blocked:
        print(f"\n  {Y}🔒 INTERCEPTED! Firewall DROPPED packet rule matching: '{res.get('attack')}'{X}")
        print(f"     Message: {res.get('message')}")
    else:
        print(f"\n  {R}⚠️ Traffic leaked through firewall.{X}")

    # ── Step 5: Clean up ──
    step(5, "Simulating Post-Incident Unblocking")
    for name in THREAT_TAXONOMY.keys():
        requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
    print(f"  {G}All firewall rules deleted. Reverted to baseline IDS mode.{X}")
    
    heading("Simulation Complete")

if __name__ == "__main__":
    main()
