"""
Main pipeline entry point for sentiment analysis
Orchestrates the entire ML pipeline from data processing to model training
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.processor import DataPreprocessor, load_data_from_url, extract_comments, create_ensemble_sentiment
from data.eda_util import plot_initial_data_exploration, plot_preprocessing_insights, plot_data_splits
from models.trainer import Vocabulary, SentimentDataset, CNNSentimentClassifier, ModelTrainer, collate_fn, device
from models.inference import evaluate_model, plot_training_curves, plot_confusion_matrix, predict_sentiment

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime

class SentimentAnalysisPipeline:
    """Complete sentiment analysis pipeline"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.results = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        print("🚀 Sentiment Analysis Pipeline Initialized")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def get_default_config(self):
        """Default configuration for the pipeline"""
        return {
            'data_url': "https://storage.googleapis.com/sachin_data1/_Social%20Media%20Analytics%20-%20LLM%20-%20Socila%20Media%20Analytics.csv",
            'model_params': {
                'embed_dim': 64,
                'num_filters': 50,
                'filter_sizes': [3, 4, 5],
                'dropout': 0.5,
                'lr': 0.001,
                'num_epochs': 15,
                'patience': 5,
                'batch_size': 32
            },
            'vocab_params': {
                'min_freq': 2,
                'max_vocab_size': 5000
            },
            'max_seq_length': 100,
            'test_size': 0.3,
            'val_size': 0.2,
            'random_state': 42
        }
    
    def run_pipeline(self, skip_training=False, skip_evaluation=False):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING SENTIMENT ANALYSIS PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load and process data
            self.load_and_process_data()
            
            # Step 2: Prepare ML data
            self.prepare_ml_data()
            
            # Step 3: Train model (if not skipping)
            if not skip_training:
                self.train_model()
            else:
                self.load_existing_model()
            
            # Step 4: Evaluate model (if not skipping)
            if not skip_evaluation:
                self.evaluate_model()
            
            # Step 5: Save results
            self.save_results()
            
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_and_process_data(self):
        """Load and process the data"""
        print("\n📊 Step 1: Loading and Processing Data")
        print("-" * 40)
        
        # Load data
        self.df = load_data_from_url(self.config['data_url'])
        if self.df is None:
            raise Exception("Failed to load data")
        
        # Find comment columns
        comment_cols = [f'Comment {i}' for i in range(1, 11)]
        self.existing_comment_cols = [col for col in comment_cols if col in self.df.columns]
        
        # Process comments
        preprocessor = DataPreprocessor()
        self.comment_df = extract_comments(self.df, self.existing_comment_cols, preprocessor)
        
        if len(self.comment_df) == 0:
            raise Exception("No comments found to process")
        
        # Create ensemble sentiment
        self.comment_df['final_sentiment'] = self.comment_df.apply(create_ensemble_sentiment, axis=1)
        
        # Filter English comments
        self.english_comments = self.comment_df[self.comment_df['language'] == 'en'].copy()
        
        print(f"✅ Processed {len(self.english_comments)} English comments")
        print(f"📊 Sentiment distribution: {self.english_comments['final_sentiment'].value_counts().to_dict()}")
        
        # Save processed data
        self.english_comments.to_csv('data/processed/english_comments.csv', index=False)
        print("💾 Processed data saved to data/processed/english_comments.csv")
        
        self.results['data_processing'] = {
            'total_comments': len(self.comment_df),
            'english_comments': len(self.english_comments),
            'sentiment_distribution': self.english_comments['final_sentiment'].value_counts().to_dict()
        }
    
    def prepare_ml_data(self):
        """Prepare data for machine learning"""
        print("\n📚 Step 2: Preparing ML Data")
        print("-" * 40)
        
        if len(self.english_comments) < 30:
            raise Exception("Not enough data for ML training")
        
        # Prepare features and labels
        X = self.english_comments['cleaned_comment'].values
        y = self.english_comments['final_sentiment'].values
        
        # Create train/val/test splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config['val_size'], 
            random_state=self.config['random_state'], stratify=y_train
        )
        
        print(f"📊 Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Create vocabulary
        self.vocab = Vocabulary(X_train, **self.config['vocab_params'])
        print(f"📚 Vocabulary size: {len(self.vocab)}")
        
        # Create datasets
        max_len = self.config['max_seq_length']
        self.train_dataset = SentimentDataset(X_train, y_train, self.vocab, max_len)
        self.val_dataset = SentimentDataset(X_val, y_val, self.vocab, max_len)
        self.test_dataset = SentimentDataset(X_test, y_test, self.vocab, max_len)
        
        # Create data loaders
        batch_size = self.config['model_params']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Save vocabulary
        joblib.dump(self.vocab, 'models/vocabulary.pkl')
        print("💾 Vocabulary saved to models/vocabulary.pkl")
        
        self.results['data_preparation'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'vocab_size': len(self.vocab),
            'max_seq_length': max_len
        }
    
    def train_model(self):
        """Train the CNN model"""
        print("\n🤖 Step 3: Training CNN Model")
        print("-" * 40)
        
        # Create model
        self.model = CNNSentimentClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.config['model_params']['embed_dim'],
            output_dim=3,
            filter_sizes=self.config['model_params']['filter_sizes'],
            num_filters=self.config['model_params']['num_filters'],
            dropout=self.config['model_params']['dropout']
        )
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"🏗️ Model created with {num_params:,} parameters")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['model_params']['lr'])
        
        # Train model
        trainer = ModelTrainer(self.model, device, self.train_loader, self.val_loader)
        train_losses, val_losses, val_accuracies = trainer.train(
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=self.config['model_params']['num_epochs'],
            patience=self.config['model_params']['patience']
        )
        
        print("✅ Training completed!")
        
        # Save training history
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        joblib.dump(training_history, 'models/training_history.pkl')
        
        self.results['training'] = {
            'num_epochs': len(train_losses),
            'best_val_accuracy': max(val_accuracies),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def load_existing_model(self):
        """Load existing trained model"""
        print("\n📁 Loading existing model...")
        
        if not os.path.exists('models/best_cnn_model.pth'):
            raise Exception("No trained model found. Please run training first.")
        
        self.model = CNNSentimentClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.config['model_params']['embed_dim'],
            output_dim=3,
            filter_sizes=self.config['model_params']['filter_sizes'],
            num_filters=self.config['model_params']['num_filters'],
            dropout=self.config['model_params']['dropout']
        )
        
        self.model.load_state_dict(torch.load('models/best_cnn_model.pth'))
        print("✅ Model loaded successfully!")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n📊 Step 4: Evaluating Model")
        print("-" * 40)
        
        # Evaluate on test set
        test_results = evaluate_model(self.model, self.test_loader, device, self.train_dataset.label_map)
        
        print(f"✅ Test Results:")
        print(f"   Accuracy: {test_results['accuracy']:.4f}")
        print(f"   F1-Weighted: {test_results['f1_weighted']:.4f}")
        print(f"   F1-Macro: {test_results['f1_macro']:.4f}")
        
        self.results['evaluation'] = {
            'test_accuracy': test_results['accuracy'],
            'test_f1_weighted': test_results['f1_weighted'],
            'test_f1_macro': test_results['f1_macro']
        }
        
        # Save detailed results
        joblib.dump(test_results, 'models/test_results.pkl')
        print("💾 Test results saved to models/test_results.pkl")
    
    def save_results(self):
        """Save pipeline results"""
        self.results['pipeline_info'] = {
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        with open('models/pipeline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("💾 Pipeline results saved to models/pipeline_results.json")
    
    def predict_text(self, text):
        """Predict sentiment for a single text"""
        if not hasattr(self, 'model'):
            raise Exception("Model not loaded. Please run pipeline first.")
        
        predicted_sentiment, probabilities = predict_sentiment(
            self.model, text, self.vocab, self.config['max_seq_length'], device
        )
        
        return {
            'text': text,
            'predicted_sentiment': predicted_sentiment,
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1],
                'positive': probabilities[2]
            }
        }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--predict', type=str, help='Predict sentiment for given text')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SentimentAnalysisPipeline()
    
    if args.predict:
        # Just predict for given text
        try:
            pipeline.prepare_ml_data()
            pipeline.load_existing_model()
            result = pipeline.predict_text(args.predict)
            print(f"\n🔮 Prediction Result:")
            print(f"Text: '{result['text']}'")
            print(f"Predicted: {result['predicted_sentiment']}")
            print(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    else:
        # Run full pipeline
        success = pipeline.run_pipeline(
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation
        )
        
        if success:
            print("\n🎊 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")

if __name__ == "__main__":
    main()
