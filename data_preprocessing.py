import re
import string
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import nltk
from config import Config
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DataPreprocessor:
    def __init__(self, model_name=Config.MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = Config()
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\'\"]', ' ', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_article(self, article):
        """Preprocess article text"""
        # Clean the text
        article = self.clean_text(article)
        
        # Add prefix for BART (if using BART)
        if "bart" in self.config.MODEL_NAME.lower():
            article = "summarize: " + article
        
        return article
    
    def preprocess_summary(self, summary):
        """Preprocess summary text"""
        summary = self.clean_text(summary)
        return summary
    
    def tokenize_data(self, examples):
        """Tokenize articles and summaries"""
        # Preprocess articles and summaries
        articles = [self.preprocess_article(article) for article in examples["article"]]
        summaries = [self.preprocess_summary(summary) for summary in examples["highlights"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            articles,
            max_length=self.config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            summaries,
            max_length=self.config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_and_preprocess_data(self):
        """Load and preprocess CNN/DailyMail dataset"""
        print("Loading CNN/DailyMail dataset...")
        
        # Load dataset
        dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_VERSION)
        
        # Use small subsets for faster training
        train_size = int(len(dataset["train"]) * self.config.TRAIN_SIZE)
        val_size = int(len(dataset["validation"]) * self.config.VAL_SIZE)
        test_size = int(len(dataset["test"]) * self.config.TEST_SIZE)
        
        print(f"Using {train_size} training samples, {val_size} validation samples, {test_size} test samples")
        
        # Create subsets
        train_dataset = dataset["train"].select(range(train_size))
        val_dataset = dataset["validation"].select(range(val_size))
        test_dataset = dataset["test"].select(range(test_size))
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=["article", "highlights", "id"]
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=["article", "highlights", "id"]
        )
        
        test_dataset = test_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=["article", "highlights", "id"]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_sample_data(self, num_samples=5):
        """Get sample data for demonstration"""
        print("Loading sample data...")
        dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_VERSION)
        test_data = dataset["test"].select(range(num_samples))
        
        samples = []
        for i in range(num_samples):
            sample = {
                "article": test_data[i]["article"],
                "original_summary": test_data[i]["highlights"],
                "preprocessed_article": self.preprocess_article(test_data[i]["article"]),
                "preprocessed_summary": self.preprocess_summary(test_data[i]["highlights"])
            }
            samples.append(sample)
        
        return samples

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = preprocessor.load_and_preprocess_data()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Show sample
    samples = preprocessor.get_sample_data(1)
    print("\nSample preprocessed data:")
    print("Article:", samples[0]["preprocessed_article"][:200] + "...")
    print("Summary:", samples[0]["preprocessed_summary"]) 