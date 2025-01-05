# Import libraries
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset

# Define device to use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Make sure that we are using the GPU
print(f"Device: {device}")

# Load the datasets using Pandas
economics_data = pd.read_csv("economics_abstracts.csv")
physics_data = pd.read_csv("physics_abstracts.csv")
mathematics_data = pd.read_csv("mathematics_abstracts.csv")

# Combine all datasets into one dataframe
combined_data = pd.concat([economics_data, physics_data, mathematics_data])

# Split the data into features (X) and labels (y)
X = combined_data['Abstract']
y = combined_data['Label']

# Tokenize and encode the abstracts
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Encode the inputs to add padding and truncation
encoded_inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors='pt')

# Convert labels to tensor
labels = torch.tensor(y.values)

# Define the DistilBERT model
class DistilBERTModel(nn.Module):
    def __init__(self):
        super(DistilBERTModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.fc1 = nn.Linear(768, 512)  # Additional fully connected layer
        self.relu = nn.ReLU()           # ReLU activation function
        self.dropout = nn.Dropout(0.1)  # Dropout layer
        self.batchnorm = nn.BatchNorm1d(512)  # Batch normalization

        self.fc2 = nn.Linear(512, 3)  # 3 output classes
        # Forward pass
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use the CLS token (The first token [0]) for summary
        
        x = self.fc1(last_hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batchnorm(x)  # Optional

        logits = self.fc2(x)
        return logits

# Model used
model = DistilBERTModel().to(device)
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# K-fold cross-validation
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Lists to store results
all_conf_matrices = []
all_class_reports = []
all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []


# Perform k-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}/{kf.get_n_splits()}")

    # Split data into train and validation sets
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Tokenize and encode the abstracts for train and validation sets
    train_inputs = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt')
    val_inputs = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt')

    # Convert labels to tensors
    train_labels = torch.tensor(y_train.values)
    val_labels = torch.tensor(y_val.values)

    # Create DataLoader for train and validation sets
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train the model
    epochs = 1
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # Calculate and print confusion matrix and classification report for each fold
    conf_matrix = confusion_matrix(all_true, all_preds)
    class_report = classification_report(all_true, all_preds, target_names=['Phyiscs', 'Economics', 'Mathematics'])

    print("\nConfusion Matrix:")
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Phyiscs', 'Economics', 'Mathematics'], yticklabels=['Phyiscs', 'Economics', 'Mathematics'])
    plt.show()

    print("\nClassification Report:")
    print(class_report)

    # Store results to calculate the mean confusion matrix and classification report
    all_conf_matrices.append(conf_matrix)
    all_class_reports.append(class_report)

# Calculate and print the mean confusion matrix and classification report
mean_conf_matrix = sum(all_conf_matrices)
mean_class_report = '\n'.join(all_class_reports)

print("\nMean Confusion Matrix:")
sns.heatmap(mean_conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Phyiscs', 'Economics', 'Mathematics'], yticklabels=['Phyiscs', 'Economics', 'Mathematics'])
plt.show()

print("\nMean Classification Report:")
print(mean_class_report)
