import streamlit as st
import pickle
import string
import nltk
import torch
from torch import nn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import GPT2Tokenizer, GPT2Model, AutoModel, BertTokenizerFast

# Initialize
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load Traditional ML Model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
ml_model = pickle.load(open("model.pkl", "rb"))

# GPT-2 Model Definition
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, max_seq_len: int, gpt_model_name: str):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output

# Load GPT-2 Model
gpt2_model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=2, max_seq_len=128, gpt_model_name="gpt2")
gpt2_model.load_state_dict(torch.load("gpt2-text-classifier-model-2class.pt", map_location=device))
gpt2_model.to(device)
gpt2_model.eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.padding_side = "left"
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# BERT Model Definition
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load BERT Model
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_classifier = BERT_Arch(bert_model).to(device)
bert_classifier.load_state_dict(torch.load("saved_weights.pt", map_location=device))
bert_classifier.eval()
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Functions for Predictions
def ml_predict(text):
    transformed_text = transform_text(text)
    vector_input = tfidf.transform([transformed_text])
    result = ml_model.predict(vector_input)[0]
    return "Spam" if result == 1 else "Not Spam"

def gpt2_predict(text):
    model_input = gpt2_tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_id = model_input["input_ids"].to(device)
    mask = model_input["attention_mask"].to(device)
    with torch.no_grad():
        output = gpt2_model(input_id, mask)
    predicted_class = output.argmax(dim=1).item()
    return "Spam" if predicted_class == 1 else "Not Spam"

def bert_predict(text):
    tokens = bert_tokenizer.encode_plus(
        text, max_length=25, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    with torch.no_grad():
        outputs = bert_classifier(input_ids, attention_mask)
    pred = torch.argmax(outputs, dim=1).item()
    return "Spam" if pred == 1 else "Not Spam"

# Voting Classifier
def voting_classifier(text):
    predictions = [ml_predict(text), gpt2_predict(text), bert_predict(text)]
    return max(set(predictions), key=predictions.count)


# Streamlit UI
st.title("📩 Ensemble Spam Classifier")
st.markdown(
    """
    This app uses **Machine Learning**, **GPT-2**, and **BERT** models to classify email/SMS messages as Spam or Not Spam.
    """
)
input_text = st.text_area("🔤 Enter your message:")

if st.button("🚀 Classify"):
    if input_text.strip():
        st.write("### Final Decision")
        result = voting_classifier(input_text)
        if result == "Spam":
            st.markdown(f"<p style='color:red; font-weight:bold; font-size:60px;'>🚨Spam</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:green; font-weight:bold; font-size:60px;'>✅Not Spam</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to classify.")
