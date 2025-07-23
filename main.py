"""
Main script for News Summarization Project
"""

import os
import sys
import argparse
import time
from datetime import datetime
import json

from config import Config
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def print_banner():
    print("="*70)
    print("         HABER BAŞLIKLARINDAN OTOMATİK ÖZETLEME SİSTEMİ")
    print("="*70)

def setup_environment():
    """Setup the environment and check dependencies"""
    print("Setting up environment...")
    
    config = Config()
    config.create_directories()
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available, using CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    return True

def run_full_pipeline(args):
    """Run the complete pipeline"""
    start_time = time.time()
    
    print("\n🚀 Starting full pipeline...")
    
    # 1. Data Preprocessing
    print("\n📊 STEP 1: Data Preprocessing")
    print("-" * 50)
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.load_and_preprocess_data()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # 2. Model Training
    print("\n🧠 STEP 2: Model Training")
    print("-" * 50)
    trainer = ModelTrainer()
    trained_model, eval_results = trainer.train_model(train_dataset, val_dataset)
    
    print(f"✅ Model training completed!")
    print(f"   Final ROUGE-L: {eval_results.get('eval_rougeL', 'N/A'):.4f}")
    
    # 3. Generate Sample Summaries (using the just-trained model)
    print("\n📝 STEP 3: Sample Generation")
    print("-" * 50)
    # Use the trained model directly instead of reloading
    trainer.model = trained_model.model  # Update the trainer's model with the trained one
    sample_results = trainer.generate_sample_summaries(test_dataset, num_samples=5)
    
    print(f"✅ Generated {len(sample_results)} sample summaries!")
    
    print(f"✅ Sample generation completed!")
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Results saved to: ./results/")
    print(f"Model saved to: ./saved_model/")
    print("="*70)

def run_training_only(args):
    """Run only the training phase"""
    print("\n🧠 Running training only...")
    
    # Data preprocessing
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.load_and_preprocess_data()
    
    # Training
    trainer = ModelTrainer()
    trained_model, eval_results = trainer.train_model(train_dataset, val_dataset)
    
    # Sample generation
    sample_results = trainer.generate_sample_summaries(test_dataset, num_samples=5)
    
    print("✅ Training completed!")



def generate_samples(args):
    """Generate sample summaries"""
    print(f"\n📝 Generating {args.num_samples} sample summaries...")
    
    # Check if model exists
    config = Config()
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"❌ Model not found at {config.MODEL_SAVE_PATH}")
        print("   Please train the model first using --train")
        return
    
    # Load data and model
    preprocessor = DataPreprocessor()
    _, _, test_dataset = preprocessor.load_and_preprocess_data()
    
    trainer = ModelTrainer()
    sample_results = trainer.generate_sample_summaries(test_dataset, num_samples=args.num_samples)
    
    print(f"✅ Generated {len(sample_results)} samples!")
    print(f"Results saved to: {config.OUTPUT_DIR}/sample_summaries.json")

def show_results():
    """Show existing results"""
    config = Config()
    results_dir = config.OUTPUT_DIR
    
    if not os.path.exists(results_dir):
        print("❌ No results found. Please run training first.")
        return
    
    print(f"\n📊 Results in {results_dir}:")
    print("-" * 50)
    
    # List all result files
    for file in os.listdir(results_dir):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  📄 {file} ({size} bytes)")
    
    # Show sample summary if exists
    sample_file = os.path.join(results_dir, "sample_summaries.json")
    if os.path.exists(sample_file):
        print("\n📝 Sample Summaries:")
        print("-" * 30)
        with open(sample_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            
        for i, sample in enumerate(samples[:3]):  # Show first 3
            print(f"\nSample {i+1}:")
            print(f"Original: {sample['original_summary'][:100]}...")
            print(f"Generated: {sample['generated_summary'][:100]}...")
            print(f"ROUGE-L: {sample['rouge_scores']['rougeL']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Haber Başlıklarından Otomatik Özetleme Sistemi")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full-pipeline", action="store_true", 
                      help="Run complete pipeline")
    group.add_argument("--train", action="store_true", 
                      help="Run training only")
    group.add_argument("--generate", action="store_true", 
                      help="Generate sample summaries")
    group.add_argument("--show-results", action="store_true", 
                      help="Show existing results")
    
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to generate (default: 5)")
    
    args = parser.parse_args()
    
    print_banner()
    
    if not setup_environment():
        print("❌ Environment setup failed. Please install required packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        if args.full_pipeline:
            run_full_pipeline(args)
        elif args.train:
            run_training_only(args)
        elif args.generate:
            generate_samples(args)
        elif args.show_results:
            show_results()
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 