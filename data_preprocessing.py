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
    
    def truncate_to_chars(self, text, max_chars):
        """Truncate text to maximum character count while preserving word boundaries"""
        if len(text) <= max_chars:
            return text
        
        # Find the last space before max_chars
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."
    
    def preprocess_article(self, article):
        """Preprocess article text"""
        # Clean the text
        article = self.clean_text(article)
        
        # Add prefix for BART (if using BART)
        if "bart" in self.config.MODEL_NAME.lower():
            article = "summarize: " + article
        
        return article
    
    def preprocess_summary(self, summary):
        """Preprocess summary text and ensure character limit"""
        summary = self.clean_text(summary)
        
        # Ensure summary doesn't exceed character limit
        if len(summary) > self.config.SUMMARY_MAX_CHARS:
            summary = self.truncate_to_chars(summary, self.config.SUMMARY_MAX_CHARS)
        
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
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training set: {len(train_dataset)} samples")
        print(f"   Validation set: {len(val_dataset)} samples")
        print(f"   Test set: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_sample_data(self, num_samples=5):
        """Get sample data for demonstration with character limits applied"""
        print("Loading sample data...")
        dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_VERSION)
        test_data = dataset["test"].select(range(num_samples * 2))  # Load more to find suitable articles
        
        samples = []
        articles_found = 0
        
        print(f"📏 Finding articles with suitable length for {self.config.TEST_ARTICLE_MAX_CHARS} character limit...")
        
        for i in range(len(test_data)):
            if articles_found >= num_samples:
                break
                
            original_article = test_data[i]["article"]
            original_summary = test_data[i]["highlights"]
            
            # Preprocess
            preprocessed_article = self.preprocess_article(original_article)
            preprocessed_summary = self.preprocess_summary(original_summary)
            
            # Apply character limits
            article_for_test = self.truncate_to_chars(original_article, self.config.TEST_ARTICLE_MAX_CHARS)
            summary_for_test = self.truncate_to_chars(original_summary, self.config.SUMMARY_MAX_CHARS)
            
            sample = {
                "article": article_for_test,
                "original_summary": summary_for_test,
                "preprocessed_article": self.truncate_to_chars(preprocessed_article, self.config.TEST_ARTICLE_MAX_CHARS),
                "preprocessed_summary": preprocessed_summary,
                "original_article_chars": len(original_article),
                "truncated_article_chars": len(article_for_test),
                "original_summary_chars": len(original_summary),
                "truncated_summary_chars": len(summary_for_test),
                "was_article_truncated": len(original_article) > self.config.TEST_ARTICLE_MAX_CHARS,
                "was_summary_truncated": len(original_summary) > self.config.SUMMARY_MAX_CHARS
            }
            
            samples.append(sample)
            articles_found += 1
            
            print(f"   Sample {articles_found}: Article {sample['truncated_article_chars']} chars" + 
                  f" {'(truncated)' if sample['was_article_truncated'] else ''}, " +
                  f"Summary {sample['truncated_summary_chars']} chars" +
                  f" {'(truncated)' if sample['was_summary_truncated'] else ''}")
        
        print(f"✅ Prepared {len(samples)} test samples with character limits applied")
        
        return samples
    
    def analyze_dataset_lengths(self, num_samples=100):
        """Analyze character lengths in the dataset"""
        print(f"📊 Analyzing character lengths in {num_samples} samples...")
        
        dataset = load_dataset(self.config.DATASET_NAME, self.config.DATASET_VERSION)
        test_data = dataset["test"].select(range(num_samples))
        
        article_lengths = []
        summary_lengths = []
        
        for i in range(len(test_data)):
            article_lengths.append(len(test_data[i]["article"]))
            summary_lengths.append(len(test_data[i]["highlights"]))
        
        print(f"📈 Article Character Statistics:")
        print(f"   Average: {np.mean(article_lengths):.0f} chars")
        print(f"   Median: {np.median(article_lengths):.0f} chars")
        print(f"   Min: {np.min(article_lengths)} chars")
        print(f"   Max: {np.max(article_lengths)} chars")
        print(f"   Articles > {self.config.TEST_ARTICLE_MAX_CHARS} chars: {sum(1 for x in article_lengths if x > self.config.TEST_ARTICLE_MAX_CHARS)}/{len(article_lengths)}")
        
        print(f"\n📈 Summary Character Statistics:")
        print(f"   Average: {np.mean(summary_lengths):.0f} chars")
        print(f"   Median: {np.median(summary_lengths):.0f} chars")
        print(f"   Min: {np.min(summary_lengths)} chars")
        print(f"   Max: {np.max(summary_lengths)} chars")
        print(f"   Summaries > {self.config.SUMMARY_MAX_CHARS} chars: {sum(1 for x in summary_lengths if x > self.config.SUMMARY_MAX_CHARS)}/{len(summary_lengths)}")

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    # Analyze dataset
    preprocessor.analyze_dataset_lengths()
    
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = preprocessor.load_and_preprocess_data()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Show sample
    samples = preprocessor.get_sample_data(3)
    print(f"\n📝 Sample preprocessed data:")
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Article ({sample['truncated_article_chars']} chars): {sample['article'][:100]}...")
        print(f"  Summary ({sample['truncated_summary_chars']} chars): {sample['original_summary']}") 