import os
import sys
import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from app.streamlit.chat_interface import chat_interface

# Set Streamlit page config
st.set_page_config(page_title="LAM Task Classifier", page_icon="🤖", layout="wide")

# Define the path to your preloaded model and tokenizer
MODEL_PATH = "/home/raed_basahih/lam-githup/app/streamlit/best_bert_multi_clean_data2_output_model (1).pth"

# Preload the model (assuming you already have this done)
@st.cache_resource
def load_model():
    # Assuming the model class and architecture are pre-defined
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    # Define the custom model that matches the preloaded state_dict structure
    class CustomBertForClassification(torch.nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.classifiers = torch.nn.ModuleDict({
                name: torch.nn.Linear(768, labels) for name, labels in num_labels.items()
            })

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs.pooler_output
            logits = {name: classifier(pooled_output) for name, classifier in self.classifiers.items()}
            return logits

    # Calculate num_labels from the state_dict structure
    num_labels = {name.split('.')[1]: weight.shape[0] for name, weight in state_dict.items() if 'weight' in name and 'classifiers' in name}

    # Create the model instance
    model = CustomBertForClassification(num_labels)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer, num_labels

# Load the preloaded model and tokenizer
model, tokenizer, num_labels = load_model()

# Define the label encoders (adjust these as per your dataset)
label_encoders = {
    'r_l': ['Medium', 'Low', 'High'],
    'f': ['No', 'Yes'],
    'l': ['No', 'Yes'],
    'e': ['No', 'Yes'],
    'rv': ['No', 'Yes'],
    'li': ['No', 'Yes']
}

# Pass the model, tokenizer, num_labels, and label_encoders to the chat_interface function
chat_interface(model, tokenizer, num_labels, label_encoders)
