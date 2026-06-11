# batch_process.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from detector import FewShotThreatDetector

class BatchProcessor:
    def __init__(self, model_path='checkpoints/best_model.pt'):
        self.detector = FewShotThreatDetector(model_path)
        self.results = []
        
    def load_csv_data(self, filepath):
        """Load network traffic from CSV"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def process_file(self, filepath, label_column=None):
        """Process entire CSV file"""
        df = self.load_csv_data(filepath)
        
        # Assume all columns except last are features
        if label_column:
            X = df.drop(columns=[label_column]).values
            y_true = df[label_column].values if label_column else None
        else:
            X = df.values
            y_true = None
        
        print(f"\n🔍 Processing {len(X)} flows...")
        
        start_time = time.time()
        
        for i, flow in enumerate(X):
            attack, confidence = self.detector.detect(flow)
            
            self.results.append({
                'index': i,
                'attack': attack,
                'confidence': confidence,
                'threat_detected': attack is not None,
                'true_label': y_true[i] if y_true is not None else None
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(X)} flows...")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Processing complete in {elapsed:.2f} seconds")
        print(f"   Average: {len(X)/elapsed:.1f} flows/second")
        
        return self.results
    
    def save_results(self, output_file='detection_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
        
        # Summary
        if 'true_label' in df.columns:
            # Calculate accuracy if true labels available
            # Only compare when both are not None
            mask = df['attack'].notna() & df['true_label'].notna()
            if mask.any():
                correct = (df.loc[mask, 'attack'] == df.loc[mask, 'true_label']).sum()
                total = mask.sum()
                accuracy = correct / total if total > 0 else 0
                print(f"   Accuracy: {accuracy:.2%}")
            else:
                print("   No valid comparisons for accuracy")
        
        # Attack statistics
        attack_counts = df[df['threat_detected']]['attack'].value_counts()
        print("\n📊 Detection Summary:")
        print(f"   Total flows: {df.shape[0]}")
        print(f"   Threats detected: {df['threat_detected'].sum()}")
        
        if len(attack_counts) > 0:
            print("\n   Attacks by type:")
            for attack, count in attack_counts.items():
                print(f"     • {attack}: {count}")
        else:
            print("\n   No attacks detected")
    
    def generate_report(self):
        """Generate HTML report (fixed Unicode issue)"""
        df = pd.DataFrame(self.results)
        
        # FIXED: Use utf-8 encoding and remove emoji or use HTML entities
        html = f"""
        <html>
        <head>
            <title>MetaShield Detection Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; }}
                .alert {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>&#x1F6E1; MetaShield Detection Report</h1>  <!-- HTML entity for shield -->
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <div class="stats">
                <div class="card">
                    <h3>Total Flows</h3>
                    <h2>{df.shape[0]}</h2>
                </div>
                <div class="card">
                    <h3>Threats Detected</h3>
                    <h2 class="alert">{df['threat_detected'].sum()}</h2>
                </div>
                <div class="card">
                    <h3>Detection Rate</h3>
                    <h2>{df['threat_detected'].mean():.2%}</h2>
                </div>
            </div>
            
            <h2>Detection Results</h2>
            {df.head(100).to_html()}
            
            <p><i>Showing first 100 of {df.shape[0]} results</i></p>
        </body>
        </html>
        """
        
        # FIXED: Specify utf-8 encoding when writing file
        with open('detection_report.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print("✅ HTML report saved to detection_report.html")

# Example usage
if __name__ == "__main__":
    processor = BatchProcessor()
    
    # Create sample CSV if none exists
    if not Path('sample_traffic.csv').exists():
        print("Creating sample traffic data...")
        sample_data = np.random.randn(1000, 78)
        df = pd.DataFrame(sample_data, columns=[f'feature_{i}' for i in range(78)])
        df['label'] = np.random.choice(['Benign', 'DoS', 'Probe'], 1000)
        df.to_csv('sample_traffic.csv', index=False)
        print("✅ Created sample_traffic.csv")
    
    # Process the file
    results = processor.process_file('sample_traffic.csv', label_column='label')
    
    # Save results
    processor.save_results()
    processor.generate_report()