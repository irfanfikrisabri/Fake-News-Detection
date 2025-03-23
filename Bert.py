# Import libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load datasets
train_df = pd.read_csv('/content/drive/My Drive/TYP/final_train.csv')
test_df = pd.read_csv('/content/drive/My Drive/TYP/final_test.csv')

# Check the datasets
print("Training Data:")
print(train_df.head())
print("\nTesting Data:")
print(test_df.head())

# Preprocess the data (REMOVED SOURCE COLUMN)
train_texts = train_df['content'].tolist()
train_labels = train_df['det_fake_label'].tolist()

test_texts = test_df['content'].tolist()
test_labels = test_df['det_fake_label'].tolist()

# Split training data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Print the length of the datasets
print("\nDataset Sizes:")
print(f"Training set: {len(train_texts)} samples")
print(f"Validation set: {len(val_texts)} samples")
print(f"Test set: {len(test_texts)} samples")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Tokenize the data
max_length = 512  # maximum input length

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

# Create PyTorch datasets
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
model = model.to('cuda')  # GPU if available

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training function with step-based evaluation
def train(model, train_loader, val_loader, optimizer, scheduler, epochs, eval_steps=50):
    model.train()
    best_val_accuracy = 0
    best_model_path = '/content/drive/My Drive/TYP/best_model.pth'
    output_lines = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        output_lines.append(f"\nEpoch {epoch + 1}/{epochs}")

        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Step-based evaluation
            if (step + 1) % eval_steps == 0:
                val_accuracy, val_report, val_predictions = evaluate(model, val_loader)
                print(f"Step {step + 1}: Validation Accuracy: {val_accuracy}")
                output_lines.append(f"Step {step + 1}: Validation Accuracy: {val_accuracy}")

                # Save the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with validation accuracy: {best_val_accuracy}")
                    output_lines.append(f"New best model saved with validation accuracy: {best_val_accuracy}")

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss}")
        output_lines.append(f"Training Loss: {avg_loss}")

    return output_lines, best_model_path

# Evaluation function with per-sample predictions
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels, all_texts = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            all_texts.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Real News', 'Human-Generated Fake News', 'LLM-Generated Fake News'])

    # Save per-sample predictions
    predictions_df = pd.DataFrame({
        'text': all_texts,
        'true_label': true_labels,
        'predicted_label': predictions
    })
    predictions_df.to_csv('/content/drive/My Drive/TYP/predictions.csv', index=False)

    return accuracy, report, predictions_df

# Function to evaluate subsets of the test data and return the results as a string
def evaluate_subsets(test_df, predictions_df):
    output_lines = []
    
    # Merge predictions with the original test dataframe
    test_df['predicted_label'] = predictions_df['predicted_label']

    # Evaluate by 'dataset' column
    output_lines.append("\nEvaluating by 'dataset' column:")
    for dataset_name in test_df['dataset'].unique():
        subset = test_df[test_df['dataset'] == dataset_name]
        accuracy = accuracy_score(subset['det_fake_label'], subset['predicted_label'])
        
        # Handle cases where a subset has fewer than 3 classes
        unique_classes = subset['det_fake_label'].unique()
        if len(unique_classes) == 3:
            report = classification_report(subset['det_fake_label'], subset['predicted_label'], target_names=['Real News', 'Human-Generated Fake News', 'LLM-Generated Fake News'])
        else:
            report = classification_report(subset['det_fake_label'], subset['predicted_label'], labels=unique_classes, target_names=[f'Class {c}' for c in unique_classes])
        
        output_lines.append(f"\nDataset: {dataset_name}")
        output_lines.append(f"Accuracy: {accuracy}")
        output_lines.append("Classification Report:")
        output_lines.append(report)

    # Evaluate by 'LLM Model' column
    output_lines.append("\nEvaluating by 'LLM model' column:")
    for llm_model in test_df['LLM model'].unique():
        subset = test_df[test_df['LLM model'] == llm_model]
        accuracy = accuracy_score(subset['det_fake_label'], subset['predicted_label'])
        
        # Handle cases where a subset has fewer than 3 classes
        unique_classes = subset['det_fake_label'].unique()
        if len(unique_classes) == 3:
            report = classification_report(subset['det_fake_label'], subset['predicted_label'], target_names=['Real News', 'Human-Generated Fake News', 'LLM-Generated Fake News'])
        else:
            report = classification_report(subset['det_fake_label'], subset['predicted_label'], labels=unique_classes, target_names=[f'Class {c}' for c in unique_classes])
        
        output_lines.append(f"\nLLM Model: {llm_model}")
        output_lines.append(f"Accuracy: {accuracy}")
        output_lines.append("Classification Report:")
        output_lines.append(report)

    return output_lines

# Training loop
epochs = 3
output_lines, best_model_path = train(model, train_loader, val_loader, optimizer, scheduler, epochs)

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Test the model
test_accuracy, test_report, test_predictions = evaluate(model, test_loader)
print(f"Test Accuracy: {test_accuracy}")
print("Test Report:")
print(test_report)
output_lines.append(f"\nTest Accuracy: {test_accuracy}")
output_lines.append("Test Report:")
output_lines.append(test_report)

# Evaluate subsets of the test data
subset_output = evaluate_subsets(test_df, test_predictions)
output_lines.extend(subset_output)  # Add subset evaluation results to the output

# Save the output to a text file
output_file_path = '/content/drive/My Drive/TYP/training_output.txt'
with open(output_file_path, 'w') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\nOutput saved to {output_file_path}")
print("Per-sample predictions saved to /content/drive/My Drive/TYP/predictions.csv")