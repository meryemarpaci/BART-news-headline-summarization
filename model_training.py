

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import numpy as np
from evaluate import load
import json
from datetime import datetime
from config import Config
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        print(f"Loading model: {self.config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
        
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            return_tensors="pt"
        )
        
        self.rouge_metric = load("rouge")
        
    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if not np.issubdtype(predictions.dtype, np.integer):
            predictions = np.argmax(predictions, axis=-1)

        predictions = np.array(predictions)
        labels = np.array(labels)

        predictions = np.nan_to_num(predictions, nan=0, posinf=0, neginf=0)
        labels = np.nan_to_num(labels, nan=0, posinf=0, neginf=0)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        predictions = predictions.astype(np.int64)
        labels = labels.astype(np.int64)

        vocab_size = self.tokenizer.vocab_size
        if vocab_size is None:
            vocab_size = len(self.tokenizer.get_vocab())

        predictions = np.clip(predictions, 0, vocab_size - 1)
        labels = np.clip(labels, 0, vocab_size - 1)

        try:
            batch_size = 8
            decoded_preds = []
            decoded_labels = []
            
            for i in range(0, len(predictions), batch_size):
                pred_batch = predictions[i:i+batch_size]
                label_batch = labels[i:i+batch_size]
                
                for pred_seq, label_seq in zip(pred_batch, label_batch):
                    try:
                        valid_pred_tokens = pred_seq[(pred_seq >= 0) & (pred_seq < vocab_size)]
                        valid_label_tokens = label_seq[(label_seq >= 0) & (label_seq < vocab_size)]
                        
                        decoded_pred = self.tokenizer.decode(valid_pred_tokens, skip_special_tokens=True)
                        decoded_label = self.tokenizer.decode(valid_label_tokens, skip_special_tokens=True)
                        
                        decoded_preds.append(decoded_pred.strip())
                        decoded_labels.append(decoded_label.strip())
                        
                    except Exception as e:
                        print(f"Warning: Failed to decode sequence, using empty string. Error: {e}")
                        decoded_preds.append("")
                        decoded_labels.append("")
                        
        except Exception as e:
            print(f"Error in batch decode: {e}")
            decoded_preds = [""] * len(predictions)
            decoded_labels = [""] * len(labels)

        decoded_preds = [pred if pred.strip() else "no summary" for pred in decoded_preds]
        decoded_labels = [label if label.strip() else "no summary" for label in decoded_labels]

        try:
            rouge_result = self.rouge_metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )

            result = {
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"], 
                "rougeL": rouge_result["rougeL"]
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            result = {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }
            
        return result
    
    def train_model(self, train_dataset, val_dataset):
        print("Starting model training...")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=200,
            logging_steps=100,
            save_steps=200,
            save_total_limit=3,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            learning_rate=self.config.LEARNING_RATE,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            report_to="none",
            predict_with_generate=True,
            generation_max_length=self.config.MAX_TARGET_LENGTH,
            generation_num_beams=self.config.NUM_BEAMS,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            include_inputs_for_metrics=True,
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("Training started...")
        train_result = trainer.train()
        
        print("Saving model...")
        trainer.save_model(self.config.MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(self.config.MODEL_SAVE_PATH)
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        with open(os.path.join(self.config.OUTPUT_DIR, "train_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        print("Training completed!")
        print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
        
        print("Running final evaluation...")
        eval_result = trainer.evaluate()
        
        with open(os.path.join(self.config.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
            
        print("Final evaluation results:")
        for key, value in eval_result.items():
            if key.startswith("eval_"):
                print(f"{key}: {value:.4f}")
                
        return trainer, eval_result
    
    def generate_sample_summaries(self, test_dataset, num_samples=5):
        print(f"Generating {num_samples} sample summaries...")
        
        if os.path.exists(self.config.MODEL_SAVE_PATH) and os.path.exists(os.path.join(self.config.MODEL_SAVE_PATH, "config.json")):
            print(f"✅ Loading trained model from {self.config.MODEL_SAVE_PATH}")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_SAVE_PATH)
            tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_SAVE_PATH)
        else:
            print("⚠️ Using current model (trained model not found in save path)")
            print(f"   Save path: {self.config.MODEL_SAVE_PATH}")
            model = self.model
            tokenizer = self.tokenizer
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        preprocessor = DataPreprocessor()
        original_samples = preprocessor.get_sample_data(num_samples)
        
        results = []
        
        for i in range(num_samples):
            input_ids = torch.tensor(test_dataset[i]["input_ids"]).unsqueeze(0)
            attention_mask = torch.tensor(test_dataset[i]["attention_mask"]).unsqueeze(0)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.MAX_TARGET_LENGTH,
                    min_length=self.config.MIN_TARGET_LENGTH,
                    num_beams=self.config.NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_summary = tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            rouge_scores = self.rouge_metric.compute(
                predictions=[generated_summary],
                references=[original_samples[i]["original_summary"]],
                use_stemmer=True
            )
            
            result = {
                "sample_id": i + 1,
                "original_article": original_samples[i]["article"],
                "original_summary": original_samples[i]["original_summary"],
                "generated_summary": generated_summary,
                "rouge_scores": rouge_scores
            }
            
            results.append(result)
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Original Summary: {original_samples[i]['original_summary']}")
            print(f"Generated Summary: {generated_summary}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        with open(os.path.join(self.config.OUTPUT_DIR, "sample_summaries.json"), "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results 