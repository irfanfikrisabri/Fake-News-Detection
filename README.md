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

The dataset consists of news articles systematically labeled across three categories (Real News, Human-Generated Fake News, and LLM-Generated Fake News) to address both traditional misinformation and emerging AI-generated content. It combines:

1. VLPFN: Expert-annotated human-written fake news <br/>
2. ISOT: Established benchmark for real vs. human-fake articles<br/>
3. Self-Generated: Synthetic LLM-generated fake news <br/>

The training set (n=3,900) maintains balanced classes (1,300 samples each), while the test set (n=1,108) reflects real-world distribution with 387 real, 392 human-fake, and 329 LLM-fake articles. All data is provided in CSV format within datasets/ with consistent columns.

### Data Format

The dataset files (`final_train.csv` and `final_test.csv`) contain the following structure:

| Attribute | Description |
|-----------|-------------|
| `content` | Full textual content of the news article |
| `det_fake_label` | Classification target with three categories: Real News (0), Human-Generated Fake News (1), LLM-Generated Fake News (2) |
| `source` | Origin dataset (VLPFN/ISOT/Self-Generated) |

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

fake-news-detection/ <br/>
├── datasets/ <br/>
│   ├── final_train.csv <br/>
│   └── final_test.csv <br/>
├── Bert.py <br/>
├── Roberta.py <br/>
├── Llama.py <br/>
├── Mistral.py <br/>
├── Svm.py <br/>
├── Nb.py <br/>
├── LICENSE <br/>
└── README.md <br/>

## Installation and Usage

### Requirements

Install the required dependencies using:

```
pip install torch transformers scikit-learn pandas numpy
```

### Model-Specific Requirements

| Model Category         | Key Packages                        | Recommended Version                                |
|------------------------|-------------------------------------|----------------------------------------------------|
| `Transformers (BERT/RoBERTa)` | `torch`, `transformers`              | `torch >= 1.9.0`, `transformers >= 4.21.0`          |
| `LLMs (Llama/Mistral)`        | `transformers`, `bitsandbytes`       | `transformers >= 4.30.0` (for quantization support) |
| `Traditional ML (SVM/NB)`     | `scikit-learn`, `numpy`              | `scikit-learn >= 1.0.0`, `numpy >= 1.21.0`          |
| `Data Handling`               | `pandas`                             | `pandas >= 1.3.0`                                   |


### Running the Models
Execute individual model implementations:

```
# Traditional Models
python Svm.py
python Nb.py

# Transformer Models  
python Bert.py
python Roberta.py

# Large Language Models
python Llama.py
python Mistral.py
```

### Data Loading
```
pythonimport pandas as pd

# Load training and testing datasets
train_data = pd.read_csv('datasets/final_train.csv')
test_data = pd.read_csv('datasets/final_test.csv')

# Extract content and labels
X_train = train_data['content']
y_train = train_data['det_fake_label']
X_test = test_data['content']  
y_test = test_data['det_fake_label']
```

## Technical Specifications

### Model Configurations

| Model Category     | Architecture             | Key Parameters                      | Training Details         |
|--------------------|--------------------------|-------------------------------------|---------------------------|
| `Traditional`      | TF-IDF + Classifier      | 5,000 features max                  | CPU-based training        |
| `BERT`             | 12-layer transformer     | 110M parameters, 512 seq length     | 3 epochs, lr=2e-5         |
| `RoBERTa`          | 12-layer transformer     | 125M parameters, 512 seq length     | 3 epochs, lr=2e-5         |
| `Llama-3.1-8B`     | Decoder-only transformer | 8B parameters, 8k context           | 4-bit quantization        |
| `Mistral-7B`       | Decoder-only transformer | 7B parameters, 8k context           | 4-bit quantization        |


### Hardware Requirements

GPU: CUDA-enabled GPU with minimum 8GB VRAM (recommended for transformer and LLM models) <br/>
RAM: Minimum 16GB system memory <br/>
Storage: At least 15GB free space for model weights and outputs <br/>

## Evaluation Metrics
The repository implements comprehensive evaluation metrics for multi-class classification:

Accuracy: Overall classification performance <br/>
Precision, Recall, F1-Score: Per-class and macro-averaged metrics <br/>
Confusion Matrix: Detailed breakdown of classification results <br/>
Classification Report: Comprehensive performance summary <br/>

Results are systematically saved with timestamp-based organization for reproducibility and comparison across different model runs.

## Reproducibility
All experiments utilize fixed random seeds (seed=42) to ensure consistent and reproducible results. Model checkpoints are automatically saved during training, and prediction outputs are stored with detailed metadata for result verification and analysis.

## Contributing
Contributions to improve the models, add new architectures, or enhance evaluation methodologies are welcome. Please ensure that new implementations follow the established code structure and include appropriate documentation and evaluation metrics.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
If you use this repository in your research, please consider citing:
```
bibtex
@inproceedings{fake-news-detection-2025,
  title={Detecting Fake News in the Era of Language Models},
  author={Sabri, Muhammad Irfan Fikri and Hettiarachchi, Hansi and Ranasinghe, Tharindu},
  booktitle={Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing},
  year={2025}
}
```
## Contact
For questions, issues, or collaboration opportunities, please open an issue on this repository or contact [irfanfikri80@gmail.com].
