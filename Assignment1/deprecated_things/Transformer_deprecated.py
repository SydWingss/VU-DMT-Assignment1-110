import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from rich.progress import Progress

# Define dataset
class MoodDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert features and labels to tensors
        features = torch.from_numpy(self.features[idx]).float()
        labels = torch.from_numpy(np.array(self.labels[idx])).float()
        return features, labels

# Define model
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=21, nhead=1)
        self.fc = nn.Linear(21, 1)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x)
        return x

# Load data
df = pd.read_csv('Assignment1/time_resampling/featured_time_resamping_sparse_matrix_data.csv')

# Group by id
grouped = df.groupby('id')

# Initialize model, optimizer and loss function
model = TransformerModel()
optimizer = AdamW(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()

# Training
with Progress() as progress:
    task = progress.add_task("[cyan]Training...", total=len(grouped))
    for epoch in range(50):
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for name, group in grouped:
            features = group.drop(['id', 'time'], axis=1).values
            labels = group['mood'].values

            # Shift labels one day forward to use previous day's features for prediction
            # features = features[:-1]
            # labels = labels[1:]

            # Split into training and test sets
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=False)

            # Standardize features
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)

            # Create data loaders
            train_dataset = MoodDataset(train_features, train_labels)
            test_dataset = MoodDataset(test_features, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Train model
            for iteration in range(50):
                losses = 0
                corrects = 0
                counts = 0
                for i, (inputs, labels) in enumerate(train_loader):
                    src = inputs[:-1].float()  # Add seq_len dimension
                    tgt = inputs[1:].float()  # Add seq_len dimension
                    labels = labels[1:].float()  # Add seq_len dimension

                    optimizer.zero_grad()

                    outputs = model(src, tgt).squeeze()
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    # Calculate total loss and accuracy
                    losses += loss.item()
                    # outputs和labels为list，若outputs和labels中的一项的差值小于0.1，则认为这一项预测正确
                    corrects += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels[i]) < 0.1])
                    counts += labels.size(0)
                    
                total_loss += losses
                total_correct += corrects
                total_count += counts
                
        # Print loss and accuracy for each epoch
        print(f'Epoch {epoch+1}: Loss = {total_loss/total_count}, Accuracy = {total_correct/total_count}')
        progress.update(task, advance=1)

# Save model
torch.save(model.state_dict(), 'Assignment1/transformer.ckpt')

# Test
model.load_state_dict(torch.load('Assignment1/transformer.ckpt'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        # Remove last day's features as we don't have the next day's mood value
        src = inputs[:-1].float()  # Add seq_len dimension
        tgt = inputs[1:].float()  # Add seq_len dimension
        labels = labels[1:].float()  # Add seq_len dimension

        outputs = model(src, tgt).squeeze()

        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        # correct += (outputs == labels).sum().item()
        # outputs和labels为list，若outputs和labels中的一项的差值小于0.1，则认为这一项预测正确
        correct += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels[i]) < 0.1])

print('Accuracy: %d %%' % (100 * correct / total))