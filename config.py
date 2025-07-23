"""
Configuration file for news summarization project
"""

import os

class Config:
    # Model configurations
    MODEL_NAME = "facebook/bart-base"  # Can be changed to "t5-small" 
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 150
    MIN_TARGET_LENGTH = 30
    
    # Training configurations
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data configurations
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    TRAIN_SIZE = 0.01  # Use small subset for faster training
    VAL_SIZE = 0.005
    TEST_SIZE = 0.005
    
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
    DO_SAMPLE = True
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True) 