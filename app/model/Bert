# Import necessary libraries
import os
import torch
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np

# Define the path to your dataset and save model path
DATA_PATH = '/content/FINAL_DB.json'
MODEL_SAVE_PATH = './best_multilingual3_bert_model.pth'  # Save path for the best model

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. Load and Preprocess the Dataset
def load_and_preprocess_data(data_path):
    data = pd.read_json(data_path, encoding='utf-8')  # Ensure proper encoding
    texts = data['task'].values
    label_columns = ['r_l', 'f', 'l', 'e', 'rv', 'li']
    label_encoders = {}
    for column in label_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, texts, label_encoders

data, texts, label_encoders = load_and_preprocess_data(DATA_PATH)

# 2. Split the Data into Training and Validation Sets
def split_data(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

train_data, val_data = split_data(data)

# 3. Create a Custom Dataset Class
class TaskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # No additional text normalization that might remove Arabic characters
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        labels = {key: torch.tensor(self.labels[key][idx], dtype=torch.long) for key in self.labels}
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# 4. Initialize the Tokenizer and Model (Using Multilingual BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

label_columns = ['r_l', 'f', 'l', 'e', 'rv', 'li']
n_labels = {key: len(label_encoders[key].classes_) for key in label_columns}

class BertMultiOutput(torch.nn.Module):
    def __init__(self, n_labels):
        super(BertMultiOutput, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert.config.hidden_dropout_prob = 0.2  # Increase dropout probability for regularization
        self.classifiers = torch.nn.ModuleDict({
            key: torch.nn.Linear(self.bert.config.hidden_size, n) for key, n in n_labels.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = {key: classifier(pooled_output) for key, classifier in self.classifiers.items()}
        return logits

model = BertMultiOutput(n_labels).to(device)

# 5. Create Data Loaders
def create_data_loader(data, tokenizer, max_len, batch_size):
    texts = data['task'].values
    labels = {key: data[key].values for key in label_columns}
    dataset = TaskDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

MAX_LEN = 128
BATCH_SIZE = 8

train_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
val_loader = create_data_loader(val_data, tokenizer, MAX_LEN, BATCH_SIZE)

# 6. Define the Loss Function and Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight_decay for regularization

def loss_fn(outputs, labels):
    losses = []
    for key in outputs.keys():
        loss = torch.nn.CrossEntropyLoss()(outputs[key], labels[key])
        losses.append(loss)
    total_loss = sum(losses)
    return total_loss

# 7. Training and Evaluation Functions with Classification Report
def compute_metrics(outputs, labels):
    preds = {key: torch.argmax(outputs[key], dim=1).cpu().numpy() for key in outputs.keys()}
    true_labels = {key: labels[key].cpu().numpy() for key in labels.keys()}
    return preds, true_labels

def generate_classification_reports(preds, true_labels):
    for key in preds.keys():
        print(f"\nClassification Report for {key}:")
        print(classification_report(true_labels[key], preds[key], target_names=label_encoders[key].classes_))

# Gradient accumulation steps
accumulation_steps = 4

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = {key: [] for key in label_columns}
    all_true_labels = {key: [] for key in label_columns}

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = {key: batch['labels'][key].to(device, non_blocking=True) for key in batch['labels']}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        preds, true_labels = compute_metrics(outputs, labels)
        for key in preds:
            all_preds[key].extend(preds[key])
            all_true_labels[key].extend(true_labels[key])

    return total_loss / len(data_loader), all_preds, all_true_labels

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = {key: [] for key in label_columns}
    all_true_labels = {key: [] for key in label_columns}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = {key: batch['labels'][key].to(device, non_blocking=True) for key in batch['labels']}

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds, true_labels = compute_metrics(outputs, labels)
            for key in preds:
                all_preds[key].extend(preds[key])
                all_true_labels[key].extend(true_labels[key])

    return total_loss / len(data_loader), all_preds, all_true_labels

# 8. Training the Model with Best Model Saving and Learning Rate Scheduler
EPOCHS = 5
best_val_loss = float('inf')

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')

    train_loss, train_preds, train_true_labels = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_preds, val_true_labels = eval_model(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # Generate classification reports for train and validation sets
    print("Train Classification Reports:")
    generate_classification_reports(train_preds, train_true_labels)

    print("Validation Classification Reports:")
    generate_classification_reports(val_preds, val_true_labels)

    # Scheduler step
    scheduler.step(val_loss)

    # Check if validation loss improved, if so, save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved with validation loss: {val_loss:.4f}")

# 9. Define the Prediction Function
def predict(model, tokenizer, text, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = {}
        for key in outputs:
            _, preds = torch.max(outputs[key], dim=1)
            label = label_encoders[key].inverse_transform(preds.cpu().numpy())
            predictions[key] = label[0]
    return predictions

# 10. Test the Complete Workflow
def main():
    # Example Arabic input
    user_input = "حذف جميع الملفات المؤقتة في النظام بشكل دائم"  # Arabic input
    task_id = "t001"
    action_id = "t001a001"

    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Make predictions
    predictions = predict(model, tokenizer, user_input, device)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
