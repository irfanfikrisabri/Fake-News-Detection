# Fake News Detection: Detecting Fake News in The Era of LLMs

This repository presents a comprehensive implementation of fake news detection using state-of-the-art machine learning models. The system addresses the evolving landscape of misinformation by classifying news articles into three distinct categories: Real News, Human-Generated Fake News, and LLM-Generated Fake News.

## Overview

The fake news detection framework implements six different models across three main categories to provide comprehensive analysis and comparison of detection methodologies. The models span from traditional machine learning approaches to cutting-edge transformer architectures and large language models, enabling robust evaluation of different techniques for identifying fabricated content in the digital information ecosystem.

## Model Architecture

The repository implements the following models:

### Traditional Machine Learning Models
- **Support Vector Machine (SVM)**: Utilizes TF-IDF vectorization with linear kernel optimization for high-dimensional text classification
- **Naive Bayes**: Employs multinomial distribution with conditional independence assumptions for probabilistic classification

### Transformer-Based Models  
- **BERT (bert-base-cased)**: Fine-tuned bidirectional encoder with task-specific classification head for sequence-level predictions
- **RoBERTa (roberta-base)**: Robustly optimized BERT approach with dynamic masking and improved pretraining methodology

### Large Language Models
- **Meta-Llama-3.1-8B-Instruct**: Instruction-tuned model with rotary positional embeddings and few-shot prompting capabilities
- **Mistral-7B-Instruct-v0.3**: Efficient architecture with sliding window attention and grouped-query attention mechanisms

## Dataset

The dataset consists of news articles systematically labeled across three categories to address both traditional human-generated misinformation and emerging AI-generated fake content. The training and testing datasets are provided in CSV format within the `datasets/` folder.

### Data Format

The dataset files (`final_train.csv` and `final_test.csv`) contain the following structure:

| Attribute | Description |
|-----------|-------------|
| `content` | Full textual content of the news article |
| `det_fake_label` | Classification target with three categories: Real News (0), Human-Generated Fake News (1), LLM-Generated Fake News (2) |

The training data was divided using an 80:20 stratified split with a fixed random seed of 42 to maintain reproducibility and preserve original label proportions across training and validation subsets.

## Model Implementation Details

### Traditional Models
Both SVM and Naive Bayes models utilize TF-IDF vectorization limited to the top 5,000 features to transform raw text into numerical representations. The SVM implementation employs a linear kernel with L2 regularization, while the Naive Bayes model operates under multinomial distribution assumptions with conditional feature independence.

### Transformer-Based Models
BERT and RoBERTa models were fine-tuned using the HuggingFace Transformers library with BertForSequenceClassification and RobertaForSequenceClassification respectively. Text sequences were preprocessed using appropriate tokenizers with a maximum sequence length of 512 tokens. Both models utilized AdamW optimizer with a learning rate of 2e-5, linear warmup scheduler, and training over 3 epochs with batch size of 8.

### Large Language Models
The LLM implementations leverage few-shot prompting strategies with carefully designed chat templates. Both Llama-3.1-8B and Mistral-7B models utilize 4-bit quantization for memory efficiency and deterministic sampling with temperature=0.0 for consistent outputs. The few-shot framework includes system prompts defining the classification task and three exemplary articles representing each target category.

## Few-Shot Prompting Framework

The LLM-based approach implements a structured prompting methodology as illustrated in the repository. The chat template consists of:

- **System Prompt**: Defines the three-class classification task with clear category descriptions
- **Few-Shot Examples**: Three carefully selected training examples, one for each category (Real News, Human-Generated Fake News, LLM-Generated Fake News)
- **Query Format**: Standardized input format for target article classification

This approach enables the models to leverage contextual learning without extensive fine-tuning, making them adaptable to the fake news detection task through in-context examples.

## Repository Structure

fake-news-detection/
├── datasets/
│   ├── final_train.csv
│   └── final_test.csv
├── Bert.py
├── Roberta.py
├── Llama.py
├── Mistral_model.py
├── Svm_model.py
├── Naive_bayes_model.py
├── LICENSE
└── README.md



## Installation and Usage

### Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt

Key dependencies include:

torch>=1.9.0
transformers>=4.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0

Running the Models
Execute individual model implementations:
bash# Traditional Models
python svm_model.py
python naive_bayes_model.py

# Transformer Models  
python bert_model.py
python roberta_model.py

# Large Language Models
python llama_model.py
python mistral_model.py
Data Loading
pythonimport pandas as pd

# Load training and testing datasets
train_data = pd.read_csv('datasets/final_train.csv')
test_data = pd.read_csv('datasets/final_test.csv')

# Extract content and labels
X_train = train_data['content']
y_train = train_data['det_fake_label']
X_test = test_data['content']  
y_test = test_data['det_fake_label']
Technical Specifications
Model Configurations
Model CategoryArchitectureKey ParametersTraining DetailsTraditionalTF-IDF + Classifier5,000 features maxCPU-based trainingBERT12-layer transformer110M parameters, 512 seq length3 epochs, lr=2e-5RoBERTa12-layer transformer125M parameters, 512 seq length3 epochs, lr=2e-5Llama-3.1-8BDecoder-only transformer8B parameters, 8k context4-bit quantizationMistral-7BDecoder-only transformer7B parameters, 8k context4-bit quantization
Hardware Requirements

GPU: CUDA-enabled GPU with minimum 8GB VRAM (recommended for transformer and LLM models)
RAM: Minimum 16GB system memory
Storage: At least 15GB free space for model weights and outputs

Evaluation Metrics
The repository implements comprehensive evaluation metrics for multi-class classification:

Accuracy: Overall classification performance
Precision, Recall, F1-Score: Per-class and macro-averaged metrics
Confusion Matrix: Detailed breakdown of classification results
Classification Report: Comprehensive performance summary

Results are systematically saved with timestamp-based organization for reproducibility and comparison across different model runs.
Reproducibility
All experiments utilize fixed random seeds (seed=42) to ensure consistent and reproducible results. Model checkpoints are automatically saved during training, and prediction outputs are stored with detailed metadata for result verification and analysis.
Contributing
Contributions to improve the models, add new architectures, or enhance evaluation methodologies are welcome. Please ensure that new implementations follow the established code structure and include appropriate documentation and evaluation metrics.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Citation
If you use this repository in your research, please consider citing:
bibtex@misc{fake-news-detection-2025,
  title={Fake News Detection: A Multi-Model Comparative Analysis},
  author={[Author Name]},
  year={2025},
  url={https://github.com/[username]/fake-news-detection}
}
Contact
For questions, issues, or collaboration opportunities, please open an issue on this repository or contact [email@domain.com].
