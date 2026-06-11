# test_with_real_data.py
import pandas as pd
import numpy as np
from detector import FewShotThreatDetector

def test_with_known_dataset():
    """Test your model on real cybersecurity dataset"""
    
    # Download a small sample of real data
    print("📥 Loading CIC-IDS2017 sample...")
    
    # You can download from: https://www.unb.ca/cic/datasets/ids-2017.html
    # For now, let's assume you have a small sample
    
    try:
        df = pd.read_csv('data/raw/sample_IDS.csv')
        print(f"✅ Loaded {len(df)} real network flows")
        
        # Initialize detector
        detector = FewShotThreatDetector()
        
        # Learn known attacks from the dataset
        attacks = df['Label'].unique()
        print(f"\n📚 Found {len(attacks)} attack types in data")
        
        for attack in attacks[:3]:  # Learn first 3 attack types
            # Get 3 examples of this attack
            examples = df[df['Label'] == attack].iloc[:3, :-1].values
            detector.learn_new_attack(attack, examples)
        
        # Test on remaining data
        print("\n🔍 Testing on unseen data...")
        correct = 0
        total = 0
        
        for _, row in df.iloc[100:200].iterrows():  # Test 100 samples
            features = row[:-1].values
            true_label = row['Label']
            
            detected, confidence = detector.detect(features)
            
            if detected == true_label:
                correct += 1
            total += 1
        
        print(f"\n📊 Results on real data:")
        print(f"   Accuracy: {correct/total:.2%}")
        print(f"   Samples tested: {total}")
        
    except FileNotFoundError:
        print("⚠️  No real dataset found. Download from:")
        print("   https://www.unb.ca/cic/datasets/ids-2017.html")

if __name__ == "__main__":
    test_with_known_dataset()