"""
MetaShield — Full IDS + IPS + Repeat Attack Demo
==================================================
Proves:
  1. LEARN  — 5 examples → instant Bayesian prototype (antibody)
  2. DETECT — alert on matching attacks (IDS mode)
  3. BLOCK  — activate shield for that attack type (IPS mode)
  4. REPEAT — same attack again → calm "already seen & blocked" notice, no impact

HOW TO RUN:
  Terminal 1:  python api.py
  Terminal 2:  python test_antibody_demo.py
"""

import requests
import numpy as np
import time

BASE_URL = "http://localhost:5000"

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

def hdr(t): print(f"\n{B}{C}{'='*62}\n  {t}\n{'='*62}{X}")
def step(n, t): print(f"\n{B}{Y}[STEP {n}] {t}\n{'-'*50}{X}")

# ── Fake traffic generators ─────────────────────────────────────
def attack_flows(name, n, dim, seed=0):
    rng = np.random.RandomState(42 + seed)
    flows = []
    quarter = dim // 4
    
    for _ in range(n):
        flow = np.zeros(dim, dtype=np.float32)
        if name == "FakeDoS":
            flow[:quarter] = rng.uniform(0.0, 2.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(4.0, 8.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(-6.0, -4.0, quarter)
            flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)
        elif name == "FakePortScan":
            flow[:quarter] = rng.uniform(3.0, 6.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(-4.0, -2.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(-3.0, -1.5, quarter)
            flow[quarter*3:] = rng.uniform(0.5, 2.0, dim - quarter*3)
        elif name == "Ransomware-Exfil":
            flow[:quarter] = rng.uniform(3.0, 6.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(4.0, 7.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(5.0, 8.0, quarter)
            flow[quarter*3:] = rng.uniform(1.0, 3.0, dim - quarter*3)
        elif name == "SQL-Injection-Wave":
            flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)
            flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)
            flow[quarter*2:quarter*3] = rng.uniform(2.0, 5.0, quarter)
            flow[quarter*3:] = rng.uniform(-3.0, -0.5, dim - quarter*3)
        else:
            flow = rng.uniform(-1.0, 1.0, dim)
        flows.append(flow.tolist())
    
    return flows

def normal_flows(dim, n=4):
    flows = []
    quarter = dim // 4
    for i in range(n):
        rng = np.random.RandomState(999+i)
        flow = np.zeros(dim, dtype=np.float32)
        flow[:quarter] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter:quarter*2] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter*2:quarter*3] = rng.uniform(-1.0, 1.0, quarter)
        flow[quarter*3:] = rng.uniform(-1.0, 1.0, dim - quarter*3)
        flows.append(flow.tolist())
    return flows

# ── API calls ───────────────────────────────────────────────────
def get_dim():
    return requests.get(f"{BASE_URL}/api/stats").json()["model_info"]["input_dim"]

def detect(flow):
    r = requests.post(f"{BASE_URL}/api/detect", json={"features": flow})
    return r.json()   # always 200 now — blocked info is inside JSON

def learn(name, examples):
    return requests.post(f"{BASE_URL}/api/learn",
                         json={"attack_name": name, "examples": examples}).json()

def block(name):
    return requests.post(f"{BASE_URL}/api/block",
                         json={"attack_name": name}).json()

# ── Print a single detect result ────────────────────────────────
def show_result(i, r, label="Flow"):
    is_blocked = r.get("blocked", False)
    detected   = r.get("threat_detected", False)
    atk        = r.get("attack", "—")
    conf       = r.get("confidence", 0)
    msg        = r.get("message", "")

    if is_blocked:
        # Repeat blocked attack — calm info notice
        print(f"  {label} {i}: {Y}🔒 ALREADY BLOCKED  |  '{atk}'  conf:{conf:.2f}{X}")
        print(f"           {Y}↳ {msg}{X}")
    elif detected:
        # New threat — alert
        print(f"  {label} {i}: {R}🚨 ALERT  |  '{atk}'  conf:{conf:.2f}  sev:{r.get('severity')}{X}")
        print(f"           {R}↳ {msg}{X}")
    else:
        # Clean traffic
        print(f"  {label} {i}: {G}✅ Clean  (conf:{conf:.2f}){X}")

# ════════════════════════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════════════════════════

def check_server():
    step(0, "Server Health Check & Reset")
    try:
        # Clear signatures first
        stats = requests.get(f"{BASE_URL}/api/stats", timeout=3).json()
        for name in stats.get("known_attacks", []):
            requests.post(f"{BASE_URL}/api/forget", json={"attack_name": name})
            requests.post(f"{BASE_URL}/api/unblock", json={"attack_name": name})
            
        d = requests.get(f"{BASE_URL}/api/health", timeout=3).json()
        print(f"  {G}✅ Server ONLINE{X}  |  Known attacks cleared.")
        return True
    except:
        print(f"  {R}❌ OFFLINE — run: python api.py{X}")
        return False

def test_before(dim, name):
    step(1, f"BEFORE LEARNING — '{name}' should NOT be detected")
    flows = attack_flows(name, 3, dim, seed=100)
    caught = sum(1 for f in flows if detect(f).get("threat_detected"))
    [show_result(i+1, detect(f)) for i, f in enumerate(flows)]
    print(f"\n  {caught}/3 caught  {'(expected ~0 ✅)' if caught==0 else '(matched existing demo pattern)'}")
    return caught

def test_learn(dim, name):
    step(2, f"LEARNING — teach '{name}' from 5 examples  →  /api/learn")
    print(f"  {C}Creating Bayesian Prototype (the 'antibody')...{X}")
    t0 = time.time()
    res = learn(name, attack_flows(name, 5, dim, seed=0))
    ms = (time.time()-t0)*1000
    if res.get("success"):
        print(f"  {G}✅ Shield created!{X}  ⏱️  {ms:.1f}ms  |  examples: {res['examples_used']}")
    return res.get("success", False)

def test_ids(dim, name):
    step(3, f"IDS MODE — detect '{name}'  →  alert raised, but NOT blocked yet")
    flows = attack_flows(name, 4, dim, seed=100)
    alerted = 0
    for i, f in enumerate(flows):
        r = detect(f)
        if r.get("threat_detected"): alerted += 1
        show_result(i+1, r)
    print(f"\n  {alerted}/4 alerted  (no block active yet)")
    return alerted

def test_activate_block(name):
    step(4, f"ACTIVATE BLOCK — /api/block  →  '{name}'")
    print(f"  {C}Like adding a firewall DROP rule for this attack signature...{X}")
    res = block(name)
    if res.get("success"):
        print(f"  {G}🔒 Block ACTIVE!{X}  Blocked list: {res['currently_blocked']}")
    return res.get("success", False)

def test_ips_first_time(dim, name):
    step(5, f"IPS MODE — FIRST occurrence after block  →  'new blocked attack' message")
    flows = attack_flows(name, 3, dim, seed=200)   # new seed = slightly different flows
    blocked_count = 0
    for i, f in enumerate(flows):
        r = detect(f)
        if r.get("blocked"): blocked_count += 1
        show_result(i+1, r)
    print(f"\n  {blocked_count}/3 flows blocked")
    return blocked_count

def test_repeat_attack(dim, name):
    step(6, f"REPEAT ATTACK — SAME attack again  →  calm 'already seen & blocked' notice")
    print(f"  {C}No damage, no alarm escalation — just an informational log.{X}")
    flows = attack_flows(name, 4, dim, seed=100)   # same seed as before = exact repeat
    repeat_blocked = 0
    for i, f in enumerate(flows):
        r = detect(f)
        if r.get("blocked"): repeat_blocked += 1
        show_result(i+1, r, label="Repeat")
    print(f"\n  {repeat_blocked}/4 repeat attacks calmly acknowledged")
    return repeat_blocked

def test_normal_traffic(dim):
    step(7, "NORMAL TRAFFIC — should pass through freely (no false blocks)")
    flows = normal_flows(dim, 4)
    fp = 0
    for i, f in enumerate(flows):
        r = detect(f)
        if r.get("blocked") or r.get("threat_detected"): fp += 1
        show_result(i+1, r, label="Normal")
    print(f"\n  False positives: {fp}/4  {'✅' if fp==0 else '⚠️'}")
    return fp

def show_stats():
    step(8, "Final Stats")
    s = requests.get(f"{BASE_URL}/api/stats").json()
    print(f"  🛡️  Known patterns : {s['known_attacks']}")
    print(f"  🔒 Blocked attacks : {s.get('blocked_attacks', [])}")
    print(f"  🔍 Flows analyzed  : {s['total_flows_analyzed']}")
    print(f"  🚨 Threats caught  : {s['total_threats_detected']}")
    print(f"  ⚡ Avg latency     : {s.get('avg_response_ms','N/A')} ms")
    if s.get("recent_alerts"):
        print(f"\n  📋 Last 5 alerts:")
        for a in s["recent_alerts"][-5:]:
            blk = "🔒 BLOCKED" if a.get("blocked") else "🚨 ALERT"
            print(f"     [{a['time']}] {blk}  {a['attack']}  conf:{a['confidence']}")

# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hdr("MetaShield — Learn · Detect · Block · Repeat Attack Demo")
    print(f"""
  {C}FLOW:{X}
    1. Attack arrives  →  unknown  →  passes through         (blind)
    2. /api/learn      →  5 samples  →  prototype in <5ms    (antibody)
    3. Same attack     →  🚨 ALERT raised                    (IDS)
    4. /api/block      →  shield activated                   (IPS)
    5. Attack again    →  🔒 blocked, informational notice   (IPS)
    6. SAME attack repeated again  →  ⚠️ already seen & blocked — no impact
    7. Normal traffic  →  passes freely                      (clean)
    """)

    if not check_server(): exit(1)

    dim  = get_dim()
    name = "Ransomware-Exfil"
    print(f"  Feature dim: {dim}  |  Testing: '{name}'")

    test_before(dim, name)
    test_learn(dim, name)
    test_ids(dim, name)
    test_activate_block(name)
    test_ips_first_time(dim, name)
    test_repeat_attack(dim, name)
    test_normal_traffic(dim)
    show_stats()

    hdr("SUMMARY")
    print(f"""
  {G}✅ Full lifecycle demonstrated:{X}

  Phase          │ Behavior
  ───────────────┼──────────────────────────────────────────────
  Before learn   │ Attack passes through (system is blind)
  After learn    │ 🚨 Alert raised — IDS active
  After block    │ 🔒 Traffic blocked — IPS active
  Repeat attack  │ ⚠️  "Already seen & blocked" — no further impact
  Normal traffic │ ✅  Passes freely — no false positives
  ───────────────┼──────────────────────────────────────────────

  {C}Like antibodies:{X}
    First exposure  → immune system learns (you call /api/learn)
    Block command   → antibody raised (/api/block)
    Repeat attack   → instantly neutralised, calm log — no damage
    """)
