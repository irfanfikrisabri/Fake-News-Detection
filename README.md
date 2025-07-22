# Fake News Detection: Detecting Fake News in The Era of LLMs
A comprehensive implementation of state-of-the-art models for detecting fake news across three categories: Real News, Human-Generated Fake News, and LLM-Generated Fake News.

# Overview
This repository provides implementations of six different model architectures for fake news detection, spanning traditional machine learning models, transformer-based models, and large language models with few-shot prompting approaches.
Model Categories

Transformer-Based Models: BERT, RoBERTa
Large Language Models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
Traditional Models: SVM, Naive Bayes

# Dataset
The models are trained and evaluated on a three-class classification task:

Class 0: Real News
Class 1: Human-Generated Fake News
Class 2: LLM-Generated Fake News

Dataset files are located in the datasets/ folder. See the datasets README for detailed information.

# Installation

Clone the repository:

bashgit clone https://github.com/irfanfikrisabri/fake-news-detection.git
cd fake-news-detection

Install dependencies:

bashpip install -r requirements.txt

For LLM models, ensure you have sufficient GPU memory (recommended: 16GB+ VRAM)

Usage
Traditional Models
pythonfrom models.traditional.svm_classifier import SVMClassifier
from models.traditional.naive_bayes_classifier import NaiveBayesClassifier

# Initialize and train SVM
svm_model = SVMClassifier()
svm_model.train('datasets/final_train.csv')
predictions = svm_model.predict('datasets/final_test.csv')
Transformer Models
pythonfrom models.transformers.bert_model import BERTClassifier
from models.transformers.roberta_model import RoBERTaClassifier

# Initialize BERT model
bert_model = BERTClassifier(model_name='bert-base-cased')
bert_model.fine_tune('datasets/final_train.csv', epochs=3, lr=2e-5)
predictions = bert_model.predict('datasets/final_test.csv')
Large Language Models
pythonfrom models.llms.llama_few_shot import LlamaFewShot
from models.llms.mistral_few_shot import MistralFewShot

# Initialize Llama with few-shot prompting
llama_model = LlamaFewShot(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
predictions = llama_model.predict_with_few_shot('datasets/final_test.csv')

# Model Specifications

Transformer Models:

Architecture: 12-layer bidirectional transformer encoders
Input: Maximum 512 tokens with padding
Optimizer: AdamW with learning rate 2e-5
Training: 3 epochs with linear warmup scheduler

Large Language Models:

Context Window: 8K tokens
Quantization: 4-bit for memory efficiency
Prompting: Few-shot with 3 examples per class
Sampling: Deterministic (temperature=0.0)

Traditional Models:

TF-IDF Vectorization: Max features = 5,000
SVM: Linear kernel with L2 regularization
Naive Bayes: Multinomial distribution assumption

# Evaluation
All models are evaluated using:

Accuracy
Precision, Recall, F1-score (macro and weighted)
Confusion Matrix
Classification Report

pythonfrom utils.evaluation_metrics import evaluate_model

# Evaluate any model
results = evaluate_model(y_true, y_pred, class_names=['Real', 'Human Fake', 'LLM Fake'])

# Reproducibility
To ensure reproducible results:

Fixed random seeds (seed=42) for data splits
Deterministic sampling for LLMs
Saved model checkpoints and configurations
Detailed hyperparameter documentation

# Requirements
Core Dependencies

Python >= 3.8
PyTorch >= 1.9.0
Transformers >= 4.21.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0

For GPU Acceleration

CUDA >= 11.0
torch with CUDA support

For LLM Models

accelerate >= 0.20.0
bitsandbytes >= 0.39.0

See requirements.txt for complete dependency list.

# Performance
Model CategoryModelAccuracyF1-Score (Macro)TraditionalSVM0.XXX0.XXXTraditionalNaive Bayes0.XXX0.XXXTransformerBERT0.XXX0.XXXTransformerRoBERTa0.XXX0.XXXLLMLlama-3.1-8B0.XXX0.XXXLLMMistral-7B0.XXX0.XXX
Note: Fill in actual performance metrics after model evaluation

# Contributing

Fork the repository
Create a feature branch (git checkout -b feature/new-model)
Commit your changes (git commit -am 'Add new model implementation')
Push to the branch (git push origin feature/new-model)
Create a Pull Request

# Citation
If you use this code in your research, please cite:
bibtex@misc{fake-news-detection-2025,
  title={Detecting Fake News in The Era of
LLMs},
  author={Muhammad Irfan Fikri Sabri},
  year={2025},
  howpublished={\url{https://github.com/irfanfikrisabri/fake-news-detection}}
}
 
# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

HuggingFace for providing pre-trained models and transformers library
Meta AI for Llama models
Mistral AI for Mistral models
The open-source community for various tools and libraries

# Contact
For questions or collaborations:

Email: irfanfikri80@gmail.com

Note: Ensure you have appropriate computational resources when running LLM models. GPU with at least 16GB VRAM is recommended for optimal performance.
