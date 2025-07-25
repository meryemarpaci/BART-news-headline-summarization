"""
Configuration file for news summarization project
"""

import os

class Config:
    # Model configurations
    MODEL_NAME = "facebook/bart-base"  
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 64  # ~256 characters (4 chars per token average)
    MIN_TARGET_LENGTH = 10
    
 
    BATCH_SIZE = 80  # Eğitim sonucunda optimize edildi
    LEARNING_RATE = 3e-5  # Slightly reduced for better convergence
    NUM_EPOCHS = 5  # Eğitim sonucunda 5 epoch kullanıldı
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data configurations
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    TRAIN_SIZE = 0.02  # 5,742 eğitim örneği
    VAL_SIZE = 0.005   # 66 doğrulama örneği
    TEST_SIZE = 0.005  # 57 test örneği
    
    # Paths
    OUTPUT_DIR = "./results"
    MODEL_SAVE_PATH = "./saved_model"
    LOGS_DIR = "./logs"
    
    # Evaluation
    ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
    
    # Device
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Generation parameters
    NUM_BEAMS = 4
    DO_SAMPLE = False  # For more consistent results
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Logging configurations - Eğitim sonuçlarına göre ayarlandı
    SAVE_STEPS = 100  # Save model more frequently
    EVAL_STEPS = 100  # Evaluate more frequently (100, 200, 300. adımlarda)
    LOGGING_STEPS = 50  # Log more frequently
    
    # Character limits for testing - Eğitim sonuçlarına göre doğrulandı
    TEST_ARTICLE_MAX_CHARS = 512  # Max characters for test articles
    SUMMARY_MAX_CHARS = 256  # Max characters for summaries
    
    # Eğitim sonuçları
    FINAL_TRAINING_LOSS = 3.4359
    FINAL_EVAL_LOSS = 4.8902
    FINAL_ROUGE_L = 0.2362
    TRAINING_TIME_MINUTES = 7.6
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True) 
