import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from rich.progress import Progress

# Load data
pd.set_option('display.max_columns', None)
df = pd.read_csv('Assignment1/time_resampling/featured_time_resamping_sparse_matrix_data.csv')

# Define dataset
class MoodDataset(Dataset):
    def __init__(self, features_1, features_2, labels):
        self.features_1 = features_1
        self.features_2 = features_2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert features and labels to tensors
        features_1 = torch.from_numpy(self.features_1[idx]).float()
        features_2 = torch.from_numpy(self.features_2[idx]).float()
        labels = torch.from_numpy(np.array(self.labels[idx])).float()
        return features_1, features_2, labels
    
# Define model
class TransformerModel(nn.Module):
    def __init__(self, num_features=21):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=num_features, nhead=1)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x)
        return x
    
# Group by id
grouped = df.groupby('id')

# Data preparation
features_1, features_2, labels = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for name, group in grouped:
    feature_1 = group.drop(['id', 'time'], axis=1)
    feature_2 = group.drop(['id', 'time'], axis=1)
    label = group['mood']

    feature_1 = feature_1.iloc[:-1].reset_index(drop=True)
    feature_2 = feature_2.iloc[1:].reset_index(drop=True)
    label = label.iloc[1:].reset_index(drop=True)
    
    # 添加feature到features，将label到labels
    features_1 = pd.concat([features_1, feature_1])
    features_2 = pd.concat([features_2, feature_2])
    labels = pd.concat([labels, label])


model = TransformerModel()
optimizer = AdamW(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()

# Split into training and test sets
train_features_1, test_features_1, train_features_2, test_features_2, train_labels, test_labels = train_test_split(features_1.values, features_2.values, labels.values, test_size=0.2, shuffle=False)

# Standardize features
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# 将features和labels的dtype转换为float
train_features_1, test_features_1, train_features_2, test_features_2, train_labels, test_labels = train_features_1.astype(np.float32), test_features_1.astype(np.float32), train_features_2.astype(np.float32), test_features_2.astype(np.float32), train_labels.astype(np.float32), test_labels.astype(np.float32)

# Create data loaders
train_dataset = MoodDataset(train_features_1, train_features_2, train_labels)
test_dataset = MoodDataset(test_features_1, test_features_2, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train model
for epoch in range(50):
    total_loss = 0
    total_correct = 0
    total_count = 0

    for i, (features_1, features_2, labels) in enumerate(train_loader):
        src = features_1.float()
        tgt = features_2.float()
        labels = labels.float()

        optimizer.zero_grad()

        outputs = model(src, tgt).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Calculate total loss and accuracy
        total_loss += loss.item()
        # total_correct += (outputs.argmax(dim=1) == tgt).sum().item()
        total_count += tgt.size(0)
    
    # Print loss and accuracy for each epoch
    print(f'Epoch {epoch+1}: Loss = {total_loss/total_count}, Accuracy = {total_correct/total_count}')

# Save model
torch.save(model.state_dict(), 'transformer.ckpt')