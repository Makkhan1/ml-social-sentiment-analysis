#!/usr/bin/env python3
"""
Complete main pipeline for CNN sentiment analysis.
Mahtab's integration script.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

print("🚀 CNN Sentiment Analysis - Mahtab's Main Pipeline")
print("=" * 60)

class MLPipeline:
    """Main ML Pipeline orchestrator."""
    
    def __init__(self):
        self.data = None
        self.model = None
        
    def step1_data_processing(self, data_path="data/raw/data.csv"):
        """Step 1: Process the data (Sachin's work)."""
        print("\n📊 STEP 1: DATA PROCESSING (Sachin's Module)")
        print("-" * 40)
        
        try:
            from data.processor import load_and_process_data, ProcessingConfig
            
            config = ProcessingConfig(
                max_vocab_size=5000,
                max_seq_length=100,
                min_word_freq=2
            )
            
            self.data = load_and_process_data(data_path, config)
            
            print(f"✅ Data processing completed!")
            print(f"   📝 Total samples: {len(self.data['X'])}")
            print(f"   📚 Vocabulary size: {len(self.data['vocab'])}")
            
            # Show sentiment distribution
            unique, counts = np.unique(self.data['y'], return_counts=True)
            for sentiment, count in zip(unique, counts):
                print(f"   {sentiment}: {count} samples")
            
            return True
            
        except ImportError:
            print("❌ Data processing module not found!")
            print("   Waiting for Sachin to push data/processor.py")
            return False
        except Exception as e:
            print(f"❌ Data processing failed: {e}")
            return False
    
    def step2_model_training(self, model_name="mahtab_sentiment_model"):
        """Step 2: Train the model (Shiv's work)."""
        print("\n🤖 STEP 2: MODEL TRAINING (Shiv's Module)")
        print("-" * 40)
        
        if self.data is None:
            print("❌ No data available! Run data processing first.")
            return False
        
        try:
            from models.trainer import ModelTrainer, TrainConfig, ModelConfig
            
            train_config = TrainConfig(
                batch_size=32,
                num_epochs=10,  # Reduced for faster training
                learning_rate=0.001,
                patience=5
            )
            
            model_config = ModelConfig(
                embed_dim=128,
                filter_sizes=[3, 4, 5],
                num_filters=100,
                dropout=0.3
            )
            
            trainer = ModelTrainer(train_config, model_config)
            results = trainer.train(self.data['X'], self.data['y'], self.data['vocab'])
            
            # Save model
            model_path = trainer.save_model(model_name)
            
            print(f"✅ Model training completed!")
            print(f"   💾 Model saved: {model_path}")
            print(f"   📊 Test accuracy: {results['test_results']['accuracy']:.4f}")
            print(f"   📈 F1-score: {results['test_results']['f1_weighted']:.4f}")
            
            return True
            
        except ImportError:
            print("❌ Model training module not found!")
            print("   Waiting for Shiv to push models/trainer.py")
            return False
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
    
    def step3_test_inference(self, model_name="mahtab_sentiment_model"):
        """Step 3: Test inference (integration test)."""
        print("\n🎯 STEP 3: INFERENCE TESTING (Integration)")
        print("-" * 40)
        
        try:
            from models.inference import load_predictor
            
            predictor = load_predictor(model_name)
            
            # Test predictions
            test_texts = [
                "I absolutely love this product! It's amazing!",
                "This is terrible, worst experience ever.",
                "It's okay, nothing special really."
            ]
            
            print("Testing predictions:")
            for text in test_texts:
                result = predictor.predict_single(text)
                sentiment = result['predicted_sentiment']
                confidence = result['confidence']
                print(f"   '{text[:30]}...' -> {sentiment} ({confidence:.3f})")
            
            print("✅ Inference testing completed!")
            return True
            
        except ImportError:
            print("❌ Inference module not found!")
            print("   Waiting for Shiv to push models/inference.py")
            return False
        except Exception as e:
            print(f"❌ Inference testing failed: {e}")
            return False
    
    def run_full_pipeline(self, data_path="data/raw/data.csv"):
        """Run the complete pipeline."""
        print("\n🔥 RUNNING COMPLETE PIPELINE")
        print("=" * 60)
        
        # Step 1: Data Processing
        if not self.step1_data_processing(data_path):
            print("⚠️  Pipeline stopped at data processing")
            return False
        
        # Step 2: Model Training
        if not self.step2_model_training():
            print("⚠️  Pipeline stopped at model training")
            return False
        
        # Step 3: Inference Testing
        if not self.step3_test_inference():
            print("⚠️  Pipeline stopped at inference testing")
            return False
        
        print("\n🎉 COMPLETE PIPELINE SUCCESSFUL!")
        print("🚀 Ready to start API and Streamlit app!")
        return True

def create_sample_data():
    """Create sample data if it doesn't exist."""
    data_path = "data/raw/data.csv"
    
    if os.path.exists(data_path):
        return data_path
    
    print("📝 Creating sample data...")
    os.makedirs("data/raw", exist_ok=True)
    
    sample_data = {
        'Post_ID': list(range(1, 21)),
        'Platform': ['twitter', 'facebook', 'instagram'] * 7,
        'Comment 1': [
            "Love this product!", "Terrible service", "Good value for money",
            "Outstanding quality!", "Poor customer support", "Decent purchase",
            "Amazing experience", "Worst product ever", "Fair price point",
            "Excellent service", "Disappointing quality", "Satisfied customer",
            "Fantastic product", "Horrible experience", "Good quality",
            "Perfect purchase", "Terrible quality", "Great value",
            "Wonderful service", "Bad experience"
        ],
        'Comment 2': [
            "Amazing quality", "Worst experience", "Decent quality",
            "Exceeded expectations", "Broke quickly", "Works fine",
            "Love it", "Hate it", "It's okay",
            "Brilliant", "Useless", "Acceptable",
            "Perfect", "Awful", "Good",
            "Excellent", "Poor", "Nice",
            "Great", "Bad"
        ],
        'Sentiment_Score': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive',
            'negative', 'neutral', 'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral', 'positive', 'negative'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(data_path, index=False)
    print(f"✅ Sample data created: {data_path}")
    return data_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CNN Sentiment Analysis Pipeline')
    parser.add_argument('--action', choices=['test', 'process', 'train', 'infer', 'full'], 
                       default='test', help='Action to perform')
    parser.add_argument('--data_path', help='Path to data file')
    
    args = parser.parse_args()
    
    # Ensure data exists
    if not args.data_path:
        args.data_path = create_sample_data()
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    if args.action == 'test':
        print("🧪 Running basic tests...")
        pipeline.step1_data_processing(args.data_path)
        
    elif args.action == 'process':
        pipeline.step1_data_processing(args.data_path)
        
    elif args.action == 'train':
        if pipeline.step1_data_processing(args.data_path):
            pipeline.step2_model_training()
            
    elif args.action == 'infer':
        pipeline.step3_test_inference()
        
    elif args.action == 'full':
        pipeline.run_full_pipeline(args.data_path)
    
    print(f"\n👨‍💼 Mahtab's pipeline execution completed!")

if __name__ == "__main__":
    main()
