import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from rich.progress import Progress

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
    def __init__(self, d_model):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=7, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x)
        return x
    
class Data_loader:
    def __init__(self, df):
        self.df = df
        self.train_loader = None
        self.test_loader = None
        
    def data_preparation(self, batch_size=32):
        # Group by id
        grouped = self.df.groupby('id')

        features_1, features_2, labels = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for _, group in grouped:
            feature_1 = group.drop(['id', 'time'], axis=1)
            feature_2 = group.drop(['id', 'time'], axis=1)
            label = group['mood']

            feature_1 = feature_1.iloc[:-1].reset_index(drop=True)
            feature_2 = feature_2.iloc[1:].reset_index(drop=True)
            label = label.iloc[1:].reset_index(drop=True)
            
            features_1 = pd.concat([features_1, feature_1])
            features_2 = pd.concat([features_2, feature_2])
            labels = pd.concat([labels, label])

        # Split into training and test sets
        train_features_1, test_features_1, train_features_2, test_features_2, train_labels, test_labels = train_test_split(features_1.values, features_2.values, labels.values, test_size=0.2, shuffle=False)
        train_features_1, test_features_1, train_features_2, test_features_2, train_labels, test_labels = train_features_1.astype(np.float32), test_features_1.astype(np.float32), train_features_2.astype(np.float32), test_features_2.astype(np.float32), train_labels.astype(np.float32), test_labels.astype(np.float32)

        # Create data loaders
        train_dataset = MoodDataset(train_features_1, train_features_2, train_labels)
        test_dataset = MoodDataset(test_features_1, test_features_2, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_features = features_1.shape[1]
        
        return train_loader, test_loader, num_features

class Trainer:
    def __init__(self, model, device, criterion, lr, epoch):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.epoch = epoch
        
    def train(self, train_loader):
        # Train model
        self.model.to(self.device)
        self.model.train()
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=self.epoch)
            for i in range(self.epoch):
                total_loss = 0
                total_correct = 0
                total_count = 0
                for features_1, features_2, labels in train_loader:
                    features_1, features_2, labels = features_1.float().to(self.device), features_2.float().to(self.device), labels.float().squeeze().to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(features_1, features_2).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    # Calculate total loss and accuracy
                    total_loss += loss.item()
                    # outputs和labels为list，若outputs和labels中的一项的差值小于0.1，则认为这一项预测正确
                    total_correct += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels[i]) < 0.1])
                    total_count += labels.size(0)
                    
                # Print loss and accuracy for each epoch
                print(f'Epoch {i+1}: Loss = {total_loss/total_count}, Accuracy = {total_correct/total_count}')
                progress.update(task, advance=1)
        return self.model
                
    def save_model(self, path):
        # Save model
        torch.save(self.model.state_dict(), path)
         
class Tester:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def test(self, test_loader):
        # Test model
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (features_1, features_2, labels) in enumerate(test_loader):
                features_1, features_2, labels = features_1.float().to(self.device), features_2.float().to(self.device), labels.float().to(self.device)
                outputs = self.model(features_1, features_2).squeeze()
                total += labels.size(0)
                correct += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels[i]) < 0.1])
        
        print('Accuracy: %d %%' % (100 * correct / total))
            
    def load_model(self, path):
        # Load model
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    criterion = nn.MSELoss()
    lr = 1e-5
    epoch = 500
    batch_size = 128
    
    # Load data
    df = pd.read_csv('Assignment1/time_resampling/featured_time_resamping_sparse_matrix_data.csv')
    data_loader = Data_loader(df)
    train_loader, test_loader, num_features = data_loader.data_preparation(batch_size)
    
    # Define model
    model = TransformerModel(d_model=num_features)
    
    trainer = Trainer(model, device, criterion, lr, epoch)
    model = trainer.train(train_loader)
    trainer.save_model('Assignment1/pytorch_model.bin')
    
    # Test model
    tester = Tester(model, device)
    tester.load_model('Assignment1/pytorch_model.bin')
    tester.test(test_loader)
    