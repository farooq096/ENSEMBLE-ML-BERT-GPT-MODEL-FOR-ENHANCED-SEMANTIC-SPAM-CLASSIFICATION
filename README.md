# ENSEMBLE ML BERT GPT MODEL FOR ENHANCED SEMANTIC SPAM CLASSIFICATION

An advanced spam classification system that leverages a hybrid ensemble of traditional Machine Learning models, BERT (Bidirectional Encoder Representations from Transformers), and GPT-based language models to achieve high accuracy in semantic spam detection. This project demonstrates the power of integrating statistical learning with transformer-based NLP models.

## 🔍 Overview

Spam detection is a classic text classification task. However, the increasing complexity and subtlety of spam messages demand more robust models. This project presents an ensemble architecture that:

- Combines multiple Machine Learning classifiers (e.g., SVM, Random Forest, Logistic Regression)
- Integrates BERT for contextual understanding
- Incorporates GPT-style embeddings for generative contextual relevance
- Utilizes ensemble learning (hard/soft voting, stacking) to combine predictions

## 📁 Project Structure

```

ENSEMBLE-ML-BERT-GPT/
├── data/                         # Dataset(s) used for training/testing
├── preprocessing/                # Data cleaning and preprocessing scripts
├── models/                       # ML, BERT, and GPT model scripts
├── ensemble/                     # Scripts for ensembling predictions
├── notebooks/                    # Jupyter notebooks for experiments and visualization
├── utils/                        # Utility functions (tokenizers, loaders, evaluators)
├── results/                      # Evaluation reports and performance metrics
├── requirements.txt              # Required dependencies
├── app.py                        # Flask/FastAPI app for demo (optional)
├── README.md                     # Project documentation
└── LICENSE

````

## 🚀 Key Features

- ✅ Preprocessing with NLTK, spaCy, and regex for cleaning and tokenization
- ✅ Fine-tuning BERT (`bert-base-uncased`) on custom spam dataset
- ✅ Embedding extraction using GPT-style models (e.g., `GPT-2`)
- ✅ Classical ML models with TF-IDF/BOW features
- ✅ Voting & Stacking ensemble techniques
- ✅ Evaluation with Accuracy, Precision, Recall, F1, AUC
- ✅ Optional web app for spam detection via REST API

## 📊 Models Used

- **Traditional ML Models**:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines
  - Random Forest

- **NLP Transformers**:
  - BERT (via Hugging Face Transformers)
  - GPT-2 embeddings (for feature augmentation)

- **Ensemble Techniques**:
  - Hard Voting
  - Soft Voting
  - Stacking (Meta-Learner: Logistic Regression or MLP)

## 🧪 Dataset

The dataset used is based on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), enriched with synthetic and real-world messages for testing semantic generalization.

| Label | Description        |
|-------|--------------------|
| ham   | Legitimate message |
| spam  | Unwanted/spam msg  |

## 📈 Evaluation Metrics

- Accuracy
- Precision / Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/ENSEMBLE-ML-BERT-GPT.git
cd ENSEMBLE-ML-BERT-GPT
pip install -r requirements.txt
````

## 🧠 Training

```bash
# Preprocess data
python preprocessing/clean_data.py

# Train ML models
python models/train_ml_models.py

# Fine-tune BERT
python models/train_bert.py

# Extract GPT embeddings
python models/extract_gpt_features.py

# Run ensemble predictions
python ensemble/ensemble_predict.py
```

## 🖥️ Optional: Run Demo Web App

```bash
# Streamlit
streamlit run app.py
```

## 📚 Dependencies

Key libraries:

* scikit-learn
* transformers
* torch
* pandas
* numpy
* nltk
* Flask or FastAPI (for deployment)

Install all with:

```bash
pip install -r requirements.txt
```

## 🔮 Future Work

* Real-time deployment using Streamlit or FastAPI
* Integration with email servers for live spam detection
* Use of LLM APIs for contextual spam flagging
* Incremental learning for evolving spam patterns

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


