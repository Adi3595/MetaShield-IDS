# api_server.py
from flask import Flask, request, jsonify
import torch
import numpy as np
from detector import FewShotThreatDetector
import threading
import time

app = Flask(__name__)

# Initialize detector globally
detector = FewShotThreatDetector()

# Store some initial attacks
initial_attacks = {
    "DoS": [np.random.randn(78) * 1.5 + 2 for _ in range(3)],
    "PortScan": [np.random.randn(78) * 0.8 - 1 for _ in range(2)],
    "BruteForce": [np.random.randn(78) * 1.2 + 0.5 for _ in range(2)]
}

for attack_name, examples in initial_attacks.items():
    detector.learn_new_attack(attack_name, examples)

@app.route('/')
def home():
    return jsonify({
        "service": "MetaShield Threat Detection API",
        "status": "active",
        "known_attacks": list(detector.known_attacks.keys()),
        "endpoints": {
            "/detect": "POST - Analyze network flow",
            "/learn": "POST - Learn new attack",
            "/stats": "GET - Get detector statistics",
            "/health": "GET - Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        "known_attacks": list(detector.known_attacks.keys()),
        "recent_alerts": list(detector.attack_history)[-10:],
        "total_alerts": len(detector.attack_history)
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Analyze a network flow"""
    data = request.json
    
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' field"}), 400
    
    features = data['features']
    if len(features) != 78:
        return jsonify({"error": f"Expected 78 features, got {len(features)}"}), 400
    
    attack, confidence = detector.detect(features)
    
    return jsonify({
        "attack": attack,
        "confidence": confidence,
        "threat_detected": attack is not None
    })

@app.route('/learn', methods=['POST'])
def learn():
    """Learn a new attack type from few examples"""
    data = request.json
    
    if 'attack_name' not in data:
        return jsonify({"error": "Missing 'attack_name' field"}), 400
    
    if 'examples' not in data:
        return jsonify({"error": "Missing 'examples' field"}), 400
    
    attack_name = data['attack_name']
    examples = data['examples']
    
    # Validate examples
    for i, ex in enumerate(examples):
        if len(ex) != 78:
            return jsonify({"error": f"Example {i} has {len(ex)} features, expected 78"}), 400
    
    # Learn the new attack
    detector.learn_new_attack(attack_name, examples)
    
    return jsonify({
        "success": True,
        "message": f"Learned new attack: {attack_name}",
        "examples_used": len(examples)
    })

@app.route('/forget', methods=['POST'])
def forget():
    """Forget an attack type"""
    data = request.json
    
    if 'attack_name' not in data:
        return jsonify({"error": "Missing 'attack_name' field"}), 400
    
    attack_name = data['attack_name']
    
    if attack_name in detector.known_attacks:
        del detector.known_attacks[attack_name]
        return jsonify({"success": True, "message": f"Forgot attack: {attack_name}"})
    else:
        return jsonify({"error": f"Attack '{attack_name}' not found"}), 404

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 Starting MetaShield API Server")
    print("="*50)
    print(f"Known attacks: {list(detector.known_attacks.keys())}")
    print(f"\nServer running at: http://localhost:5000")
    print("\nTry these commands in another terminal:")
    print("  curl http://localhost:5000/health")
    print("  curl http://localhost:5000/stats")
    print("\nTo detect a threat:")
    print("  curl -X POST http://localhost:5000/detect \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"features\": [0.1, 0.2, ...]}'")
    print("\n" + "="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)