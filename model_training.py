import os
import torch
import pandas as pd
import csv
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
import numpy as np
from evaluate import load
import json
from datetime import datetime
from config import Config
from data_preprocessing import DataPreprocessor

class LossLoggingCallback(TrainerCallback):
    """Custom callback to log training losses to CSV"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.losses = []
        self.eval_losses = []
        
        # Initialize CSV file
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'train_loss', 'eval_loss', 'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'timestamp'])
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            epoch = state.epoch
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            train_loss = logs.get('train_loss', '')
            eval_loss = logs.get('eval_loss', '')
            eval_rouge1 = logs.get('eval_rouge1', '')
            eval_rouge2 = logs.get('eval_rouge2', '')
            eval_rougeL = logs.get('eval_rougeL', '')
            
            # Only log if we have meaningful data
            if train_loss or eval_loss:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, epoch, train_loss, eval_loss, eval_rouge1, eval_rouge2, eval_rougeL, timestamp])
                
                # Print detailed log
                if train_loss:
                    print(f"📊 Step {step} | Epoch {epoch:.1f} | Train Loss: {train_loss:.4f}")
                if eval_loss:
                    print(f"📊 Step {step} | Epoch {epoch:.1f} | Eval Loss: {eval_loss:.4f} | ROUGE-L: {eval_rougeL:.4f}")

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
        
        # Create loss logging file
        self.loss_log_file = os.path.join(self.config.LOGS_DIR, f"training_losses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
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
                        
                        # Ensure summaries don't exceed character limit
                        if len(decoded_pred) > self.config.SUMMARY_MAX_CHARS:
                            decoded_pred = decoded_pred[:self.config.SUMMARY_MAX_CHARS].rsplit(' ', 1)[0] + "..."
                        
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
        print(f"📊 Training Configuration:")
        print(f"   Epochs: {self.config.NUM_EPOCHS}")
        print(f"   Batch Size: {self.config.BATCH_SIZE}")
        print(f"   Learning Rate: {self.config.LEARNING_RATE}")
        print(f"   Summary Max Length: {self.config.MAX_TARGET_LENGTH} tokens (~{self.config.SUMMARY_MAX_CHARS} chars)")
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=self.config.EVAL_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
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
            fp16=False,  # Keep as False for Colab compatibility
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            include_inputs_for_metrics=True,
            logging_dir=self.config.LOGS_DIR,
            dataloader_num_workers=0,  # Colab compatibility
        )
        
        # Create loss logging callback
        loss_callback = LossLoggingCallback(self.loss_log_file)
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5), loss_callback]
        )
        
        print("Training started...")
        print(f"📁 Loss values will be saved to: {self.loss_log_file}")
        train_result = trainer.train()
        
        print("Saving model...")
        trainer.save_model(self.config.MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(self.config.MODEL_SAVE_PATH)
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["config"] = {
            "num_epochs": self.config.NUM_EPOCHS,
            "batch_size": self.config.BATCH_SIZE,
            "learning_rate": self.config.LEARNING_RATE,
            "max_target_length": self.config.MAX_TARGET_LENGTH,
            "summary_max_chars": self.config.SUMMARY_MAX_CHARS
        }
        
        with open(os.path.join(self.config.OUTPUT_DIR, "train_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        print("Training completed!")
        print(f"📊 Final Training Metrics:")
        print(f"   Training loss: {metrics.get('train_loss', 'N/A')}")
        print(f"   Training runtime: {metrics.get('train_runtime', 0)/60:.1f} minutes")
        print(f"   Training samples per second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
        
        print("Running final evaluation...")
        eval_result = trainer.evaluate()
        
        with open(os.path.join(self.config.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(eval_result, f, indent=2)
            
        print("📊 Final evaluation results:")
        for key, value in eval_result.items():
            if key.startswith("eval_"):
                print(f"   {key}: {value:.4f}")
                
        # Create summary of training
        summary_file = os.path.join(self.config.OUTPUT_DIR, "training_summary.txt")
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.config.MODEL_NAME}\n")
            f.write(f"Epochs: {self.config.NUM_EPOCHS}\n")
            f.write(f"Batch Size: {self.config.BATCH_SIZE}\n")
            f.write(f"Learning Rate: {self.config.LEARNING_RATE}\n")
            f.write(f"Training Samples: {len(train_dataset)}\n")
            f.write(f"Validation Samples: {len(val_dataset)}\n")
            f.write(f"Final Training Loss: {metrics.get('train_loss', 'N/A')}\n")
            f.write(f"Final ROUGE-L: {eval_result.get('eval_rougeL', 'N/A'):.4f}\n")
            f.write(f"Training Time: {metrics.get('train_runtime', 0)/60:.1f} minutes\n")
            f.write(f"Loss Log File: {self.loss_log_file}\n")
        
        print(f"📄 Training summary saved to: {summary_file}")
        
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
            # Get test article and truncate to max characters
            test_article = original_samples[i]["article"]
            if len(test_article) > self.config.TEST_ARTICLE_MAX_CHARS:
                test_article = test_article[:self.config.TEST_ARTICLE_MAX_CHARS] + "..."
                print(f"📏 Article {i+1} truncated to {self.config.TEST_ARTICLE_MAX_CHARS} characters")
            
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
                    do_sample=self.config.DO_SAMPLE,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_summary = tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            # Ensure summary doesn't exceed character limit
            if len(generated_summary) > self.config.SUMMARY_MAX_CHARS:
                generated_summary = generated_summary[:self.config.SUMMARY_MAX_CHARS].rsplit(' ', 1)[0] + "..."
            
            rouge_scores = self.rouge_metric.compute(
                predictions=[generated_summary],
                references=[original_samples[i]["original_summary"]],
                use_stemmer=True
            )
            
            result = {
                "sample_id": i + 1,
                "original_article": test_article,  # Use truncated version
                "original_article_chars": len(test_article),
                "original_summary": original_samples[i]["original_summary"],
                "generated_summary": generated_summary,
                "generated_summary_chars": len(generated_summary),
                "rouge_scores": rouge_scores
            }
            
            results.append(result)
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Article ({len(test_article)} chars): {test_article[:100]}...")
            print(f"Original Summary: {original_samples[i]['original_summary']}")
            print(f"Generated Summary ({len(generated_summary)} chars): {generated_summary}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # Save results
        output_file = os.path.join(self.config.OUTPUT_DIR, "sample_summaries.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create human-readable summary
        readable_file = os.path.join(self.config.OUTPUT_DIR, "sample_summaries_readable.txt")
        with open(readable_file, "w", encoding='utf-8') as f:
            f.write("SAMPLE SUMMARIES\n")
            f.write("="*70 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Summary character limit: {self.config.SUMMARY_MAX_CHARS}\n")
            f.write(f"Article character limit: {self.config.TEST_ARTICLE_MAX_CHARS}\n\n")
            
            for result in results:
                f.write(f"SAMPLE {result['sample_id']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Article ({result['original_article_chars']} chars):\n{result['original_article']}\n\n")
                f.write(f"Original Summary:\n{result['original_summary']}\n\n")
                f.write(f"Generated Summary ({result['generated_summary_chars']} chars):\n{result['generated_summary']}\n\n")
                f.write(f"ROUGE Scores:\n")
                f.write(f"  ROUGE-1: {result['rouge_scores']['rouge1']:.4f}\n")
                f.write(f"  ROUGE-2: {result['rouge_scores']['rouge2']:.4f}\n")
                f.write(f"  ROUGE-L: {result['rouge_scores']['rougeL']:.4f}\n")
                f.write("\n" + "="*70 + "\n\n")
        
        print(f"📄 Results saved to:")
        print(f"   JSON: {output_file}")
        print(f"   Readable: {readable_file}")
            
        return results 