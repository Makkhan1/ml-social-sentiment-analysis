"""
Main pipeline orchestration for CNN sentiment analysis.
Author: Mahtab (Project Lead)
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional
import pickle
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import from team members' work (will be available after they complete)
try:
    from data.processor import load_and_process_data, ProcessingConfig
    from models.trainer import ModelTrainer, TrainConfig, ModelConfig, plot_training_curves, plot_confusion_matrix
    from models.inference import SentimentPredictor, load_predictor
except ImportError as e:
    print(f"⚠️ Some modules not available yet: {e}")
    print("This is normal if team members haven't finished their work yet.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLPipeline:
    """Complete ML pipeline for sentiment analysis."""
    
    def __init__(self, config_file: Optional[str] = None):
        try:
            self.processing_config = ProcessingConfig()
            self.train_config = TrainConfig()
            self.model_config = ModelConfig()
        except NameError:
            print("⚠️ Config classes not available yet - team members need to complete their work")
            self.processing_config = None
            self.train_config = None
            self.model_config = None
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from file."""
        logger.info(f"Config loading from {config_file} - will implement this later")
    
    def run_data_processing(self, data_path: str, save_processed: bool = True) -> Dict:
        """Run data processing pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # This will work when Sachin completes his part
            data = load_and_process_data(data_path, self.processing_config)
            
            logger.info(f"✅ Data processing completed successfully!")
            logger.info(f"   - Total samples: {len(data['X'])}")
            logger.info(f"   - Vocabulary size: {len(data['vocab'])}")
            
            # Print sentiment distribution
            unique, counts = np.unique(data['y'], return_counts=True)
            sentiment_dist = dict(zip(unique, counts))
            logger.info(f"   - Sentiment distribution: {sentiment_dist}")
            
            if save_processed:
                processed_dir = Path("data/processed")
                processed_dir.mkdir(exist_ok=True)
                
                # Save processed data
                with open(processed_dir / "processed_data.pkl", 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"   - Processed data saved to data/processed/")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {str(e)}")
            logger.error("This might be because Sachin hasn't completed the data processing module yet")
            raise
    
    def run_model_training(self, data: Dict, model_name: str = "sentiment_cnn") -> Dict:
        """Run model training pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # This will work when Shiv completes his part
            trainer = ModelTrainer(self.train_config, self.model_config)
            results = trainer.train(data['X'], data['y'], data['vocab'])
            
            # Save model
            model_path = trainer.save_model(model_name)
            
            logger.info(f"✅ Model training completed successfully!")
            logger.info(f"   - Model saved to: {model_path}")
            logger.info(f"   - Test accuracy: {results['test_results']['accuracy']:.4f}")
            logger.info(f"   - F1-weighted: {results['test_results']['f1_weighted']:.4f}")
            
            # Plot training curves (if possible)
            try:
                plot_training_curves(
                    results['train_losses'], 
                    results['val_losses'], 
                    results['val_accuracies']
                )
                
                plot_confusion_matrix(
                    results['test_results']['confusion_matrix'],
                    ['negative', 'neutral', 'positive']
                )
            except Exception as e:
                logger.warning(f"Could not display plots: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            logger.error("This might be because Shiv hasn't completed the model training module yet")
            raise
    
    def run_inference(self, texts: list, model_name: str = "sentiment_cnn") -> list:
        """Run inference pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING INFERENCE PIPELINE")
        logger.info("=" * 50)
        
        try:
            # This will work when Shiv completes his part
            predictor = load_predictor(model_name)
            results = predictor.predict_batch(texts)
            
            logger.info(f"✅ Inference completed successfully!")
            logger.info(f"   - Processed {len(texts)} texts")
            
            # Print sample results
            for i, result in enumerate(results[:3]):
                text_preview = result['text'][:50] + '...' if len(result['text']) > 50 else result['text']
                logger.info(f"   - Sample {i+1}: '{text_preview}' -> {result['predicted_sentiment']} ({result['confidence']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {str(e)}")
            logger.error("This might be because Shiv hasn't completed the inference module yet")
            raise
    
    def run_full_pipeline(self, data_path: str, model_name: str = "sentiment_cnn") -> Dict:
        """Run complete pipeline from data to trained model."""
        logger.info("🚀 STARTING COMPLETE ML PIPELINE")
        
        # Step 1: Data Processing
        data = self.run_data_processing(data_path)
        
        # Step 2: Model Training
        training_results = self.run_model_training(data, model_name)
        
        # Step 3: Sample Inference Test
        sample_texts = [
            "I love this product! It's amazing!",
            "This is terrible, worst experience ever.",
            "It's okay, nothing special really."
        ]
        
        inference_results = self.run_inference(sample_texts, model_name)
        
        logger.info("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        
        return {
            'data': data,
            'training_results': training_results,
            'sample_inference': inference_results
        }

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description='CNN Sentiment Analysis Pipeline')
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data processing command
    process_parser = subparsers.add_parser('process', help='Process data')
    process_parser.add_argument('--data_path', required=True, help='Path to input data CSV')
    process_parser.add_argument('--save', action='store_true', help='Save processed data')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data_path', help='Path to input data CSV')
    train_parser.add_argument('--processed_data', help='Path to processed data pickle')
    train_parser.add_argument('--model_name', default='sentiment_cnn', help='Model name for saving')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model_name', default='sentiment_cnn', help='Model name to load')
    infer_parser.add_argument('--text', help='Single text to analyze')
    infer_parser.add_argument('--texts_file', help='File with texts to analyze (one per line)')
    infer_parser.add_argument('--output', help='Output file for results')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--data_path', required=True, help='Path to input data CSV')
    full_parser.add_argument('--model_name', default='sentiment_cnn', help='Model name for saving')
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    try:
        if args.command == 'process':
            data = pipeline.run_data_processing(args.data_path, args.save)
            print(f"✅ Data processing completed. Vocabulary size: {len(data['vocab'])}")
        
        elif args.command == 'train':
            # Load or process data
            if args.processed_data:
                with open(args.processed_data, 'rb') as f:
                    data = pickle.load(f)
            elif args.data_path:
                data = pipeline.run_data_processing(args.data_path)
            else:
                print("❌ Either --data_path or --processed_data must be provided")
                return
            
            # Override config if provided
            if args.epochs and pipeline.train_config:
                pipeline.train_config.num_epochs = args.epochs
            if args.batch_size and pipeline.train_config:
                pipeline.train_config.batch_size = args.batch_size
            if args.learning_rate and pipeline.train_config:
                pipeline.train_config.learning_rate = args.learning_rate
            
            results = pipeline.run_model_training(data, args.model_name)
            print(f"✅ Training completed. Test accuracy: {results['test_results']['accuracy']:.4f}")
        
        elif args.command == 'infer':
            if args.text:
                texts = [args.text]
            elif args.texts_file:
                with open(args.texts_file, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                print("❌ Either --text or --texts_file must be provided")
                return
            
            results = pipeline.run_inference(texts, args.model_name)
            
            # Output results
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Results saved to {args.output}")
            else:
                for result in results:
                    print(f"Text: {result['text']}")
                    print(f"Sentiment: {result['predicted_sentiment']} (confidence: {result['confidence']:.3f})")
                    print("-" * 50)
        
        elif args.command == 'full':
            results = pipeline.run_full_pipeline(args.data_path, args.model_name)
            print("🎉 Complete pipeline finished successfully!")
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        print("\n🔍 TROUBLESHOOTING:")
        print("1. Make sure Sachin has completed the data processing module")
        print("2. Make sure Shiv has completed the model training module")
        print("3. Check that the data file exists and is in the correct format")
        sys.exit(1)

if __name__ == "__main__":
    main()
