import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['text'] = df['title'] + " " + df['description']
    return df

# Prepare dataset for question answering
def prepare_qa_dataset(texts, tokenizer, max_length=384):
    questions = ["What happened?" for _ in texts]  # Simple question for each text
    
    encodings = tokenizer(questions, texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    
    # Simulate answer positions (you should replace this with actual answer positions if available)
    start_positions = torch.tensor([1 for _ in texts])
    end_positions = torch.tensor([10 for _ in texts])
    
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "start_positions": start_positions,
        "end_positions": end_positions
    })

# Custom Question Answering Model
class CustomQuestionAnsweringModel(AutoModelForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                start_positions=None, end_positions=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            **kwargs
        )
        
        if self.training:
            return {
                'loss': outputs.loss,
                'start_logits': outputs.start_logits,
                'end_logits': outputs.end_logits
            }
        else:
            return outputs

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

# Main function to run the entire process
def main():
    # Load and preprocess data
    df = load_and_preprocess_data('news_data.csv')
    
    # Split data
    train_texts, val_texts = train_test_split(df['text'].tolist(), test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare datasets
    train_dataset = prepare_qa_dataset(train_texts, tokenizer)
    val_dataset = prepare_qa_dataset(val_texts, tokenizer)
    
    # Initialize model
    model = CustomQuestionAnsweringModel.from_pretrained("bert-base-uncased")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    model.save_pretrained("./qa_model")
    tokenizer.save_pretrained("./qa_model")
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()