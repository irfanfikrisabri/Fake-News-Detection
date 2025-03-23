# Install required libraries
!pip install scikit-learn pandas

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

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

# Preprocess the data
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

# Create a pipeline for TF-IDF vectorization and Naive Bayes classification
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Convert text to TF-IDF features
    ('nb', MultinomialNB())  # Use Multinomial Naive Bayes
])

# Train the Naive Bayes model
print("\nTraining the Naive Bayes model...")
nb_pipeline.fit(train_texts, train_labels)

# Evaluate on the validation set
val_predictions = nb_pipeline.predict(val_texts)
val_accuracy = accuracy_score(val_labels, val_predictions)
val_report = classification_report(val_labels, val_predictions, target_names=['Real News', 'Human-Generated Fake News', 'LLM-Generated Fake News'])

# Evaluate on the test set
test_predictions = nb_pipeline.predict(test_texts)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_report = classification_report(test_labels, test_predictions, target_names=['Real News', 'Human-Generated Fake News', 'LLM-Generated Fake News'])

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'text': test_texts,
    'true_label': test_labels,
    'predicted_label': test_predictions
})
predictions_df.to_csv('/content/drive/My Drive/TYP/NaiveBayes/nb_predictions.csv', index=False)

print("\nPer-sample predictions saved to /content/drive/My Drive/TYP/NaiveBayes/nb_predictions.csv")

# Function to evaluate subsets of the test data
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

# Evaluate subsets of the test data
subset_output = evaluate_subsets(test_df, predictions_df)

# Combine all results into a single list
combined_results = [
    "Validation Results:",
    f"Validation Accuracy: {val_accuracy}",
    "Validation Classification Report:",
    val_report,
    "\nTest Results:",
    f"Test Accuracy: {test_accuracy}",
    "Test Classification Report:",
    test_report
]

# Add subset evaluation results to the combined results
combined_results.extend(subset_output)

# Save the combined results to a text file
output_file_path = '/content/drive/My Drive/TYP/NaiveBayes/naive_bayes_evaluation.txt'
with open(output_file_path, 'w') as f:
    for line in combined_results:
        f.write(line + "\n")

print(f"\nCombined evaluation results saved to {output_file_path}")