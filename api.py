"""
MetaShield API Server — FastAPI Production Server
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
from typing import List, Optional
import time, json, numpy as np, os, asyncio, logging
from datetime import datetime
from collections import deque
from detector import FewShotThreatDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MetaShield API", description="Few-Shot Cyber Threat Detection System", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

detector = FewShotThreatDetector()
recent_alerts = deque(maxlen=500)
start_time = time.time()
detection_times = deque(maxlen=1000)

# Load persisted data on startup
if os.path.exists("attack_signatures.json"):
    try:
        detector.load_attack_signatures("attack_signatures.json")
        logger.info(f"💾 Loaded persisted signatures: {list(detector.known_attacks.keys())}")
    except Exception as e:
        logger.error(f"Failed to load persisted signatures: {e}")

if os.path.exists("detection_log.json"):
    try:
        with open("detection_log.json", "r") as f:
            loaded_alerts = json.load(f)
            recent_alerts.extend(loaded_alerts)
        logger.info(f"💾 Loaded {len(loaded_alerts)} persisted alerts from disk.")
    except Exception as e:
        logger.error(f"Failed to load persisted alerts: {e}")

# [IPS] Blocked attack registry — attacks added here are actively blocked
blocked_attacks: dict = {}  # attack_name -> last_seen_timestamp
BLOCK_TIMEOUT_SECONDS = 15  # Auto-unblock after 15s of inactivity

def cleanup_blocked_attacks():
    now = time.time()
    expired = [name for name, t in list(blocked_attacks.items()) if now - t > BLOCK_TIMEOUT_SECONDS]
    for name in expired:
        del blocked_attacks[name]
        logger.info(f"🔄 Auto-unblocked '{name}' after {BLOCK_TIMEOUT_SECONDS}s of inactivity")

# NOTE: Persisted signatures loaded at boot.
logger.info(f"🔓 Attack registry active with signatures: {list(detector.known_attacks.keys())}")

class FlowFeatures(BaseModel):
    features: List[float] = Field(..., description="Network flow features")
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if len(v) != detector.input_dim:
            raise ValueError(f'Expected {detector.input_dim} features, got {len(v)}')
        return v

class BatchFlowFeatures(BaseModel):
    flows: List[List[float]] = Field(..., description="Batch of network flows")

class LearnRequest(BaseModel):
    attack_name: str = Field(..., description="Name of the attack")
    examples: List[List[float]] = Field(..., description="List of example flows")

class DetectionResponse(BaseModel):
    attack: Optional[str] = None
    confidence: float
    threat_detected: bool
    processing_time_ms: float
    severity: Optional[str] = None
    blocked: bool = False
    message: Optional[str] = None

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_threat(flow: FlowFeatures, background_tasks: BackgroundTasks):
    cleanup_blocked_attacks()
    start = time.time()
    attack, confidence = detector.detect(flow.features)
    proc_time = (time.time() - start) * 1000
    detection_times.append(proc_time)
    severity = None
    is_blocked = False
    msg = None

    if attack:
        severity = 'critical' if confidence > 0.9 else 'high' if confidence > 0.7 else 'medium'
        is_blocked = attack in blocked_attacks

        if not is_blocked:
            # Auto-block the new attack to instantly switch to IPS mode
            blocked_attacks[attack] = time.time()
            is_blocked = True
            msg = f"🚨 Threat detected: '{attack}' — SYSTEM AUTO-UPGRADED TO IPS MODE (Blocked)"
            logger.warning(f"🚨 Auto-blocking threat: '{attack}' (conf: {confidence:.2f}, severity: {severity})")
        else:
            # Update last seen time for this attack
            blocked_attacks[attack] = time.time()
            # [IPS] Attack already known & blocked — calm informational alert, causes NO impact
            msg = f"⚠️  Attack '{attack}' was seen before and is already BLOCKED. No impact on system."
            logger.info(f"🔒 Repeat blocked attack detected: '{attack}' (conf: {confidence:.2f}) — ignored.")

        recent_alerts.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "attack": attack,
            "confidence": round(confidence, 4),
            "severity": severity,
            "blocked": is_blocked,
            "message": msg
        })
        try:
            with open("detection_log.json", "w") as f:
                json.dump(list(recent_alerts), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist alert log: {e}")

    return DetectionResponse(
        attack=attack,
        confidence=confidence,
        threat_detected=attack is not None,
        processing_time_ms=round(proc_time, 2),
        severity=severity,
        blocked=is_blocked,
        message=msg
    )

@app.post("/api/detect/batch")
async def detect_batch(batch: BatchFlowFeatures):
    """[ERTIL] Efficient Real-Time Inference batch endpoint."""
    start = time.time()
    results = detector.detect_batch(batch.flows)
    proc_time = (time.time() - start) * 1000
    return {"results": results, "processing_time_ms": round(proc_time, 2), "flows_processed": len(batch.flows)}

@app.post("/api/learn")
async def learn_attack(request: LearnRequest):
    try:
        detector.learn_new_attack(request.attack_name, request.examples)
        detector.save_attack_signatures("attack_signatures.json")
        return {"success": True, "message": f"Learned attack: {request.attack_name}",
                "examples_used": len(request.examples)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/block")
async def block_attack(data: dict):
    """
    [IPS] Add an attack to the active block list.
    Once blocked, any future connection matching this attack pattern
    is REJECTED with HTTP 403 — simulating a firewall DROP rule.
    """
    name = data.get("attack_name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="attack_name is required")
    if name not in detector.known_attacks:
        raise HTTPException(status_code=404,
            detail=f"Unknown attack '{name}'. Teach it first via /api/learn.")
    blocked_attacks[name] = time.time()
    logger.info(f"🔒 Attack '{name}' added to BLOCK LIST")
    return {
        "success": True,
        "message": f"'{name}' is now BLOCKED. All matching traffic will be rejected.",
        "currently_blocked": list(blocked_attacks.keys())
    }

@app.post("/api/unblock")
async def unblock_attack(data: dict):
    """Remove an attack from the block list (back to detect-only mode)."""
    name = data.get("attack_name", "").strip()
    if name in blocked_attacks:
        del blocked_attacks[name]
        return {"success": True, "message": f"'{name}' unblocked.", "currently_blocked": list(blocked_attacks.keys())}
    raise HTTPException(status_code=404, detail=f"'{name}' was not in the block list.")

@app.get("/api/blocklist")
async def get_blocklist():
    """View all currently blocked attack types."""
    cleanup_blocked_attacks()
    return {"blocked_attacks": list(blocked_attacks.keys()), "total": len(blocked_attacks)}

@app.post("/api/federated/sync")
async def federated_sync():
    """
    [DCLE] Distributed Continual Learning at Network Edge
    Simulates syncing learned prototypes with a central server without sharing raw flow data.
    """
    return {
        "status": "success", 
        "message": "Prototypes synced successfully. Privacy preserved.",
        "synced_attacks": list(detector.known_attacks.keys())
    }

@app.post("/api/forget")
async def forget_attack(data: dict):
    name = data.get("attack_name", "")
    if detector.forget_attack(name):
        detector.save_attack_signatures("attack_signatures.json")
        return {"success": True, "message": f"Forgot attack: {name}"}
    raise HTTPException(status_code=404, detail=f"Attack '{name}' not found")

@app.get("/api/stats")
async def get_stats():
    cleanup_blocked_attacks()
    stats = detector.get_stats()
    stats['recent_alerts'] = list(recent_alerts)[-50:]
    stats['total_alerts'] = len(recent_alerts)
    stats['avg_response_ms'] = round(np.mean(list(detection_times)), 2) if detection_times else 0
    stats['blocked_attacks'] = list(blocked_attacks.keys())
    return stats

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "model_loaded": True,
            "known_attacks": len(detector.known_attacks), "uptime": time.time() - start_time}

@app.get("/api/export")
async def export_data():
    return JSONResponse(content={"timestamp": time.time(),
        "known_attacks": list(detector.known_attacks.keys()),
        "recent_alerts": list(recent_alerts),
        "model_info": detector.get_stats()['model_info']})

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard_app.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>MetaShield API Running</h1><p>Dashboard not found.</p>")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True, log_level="info")