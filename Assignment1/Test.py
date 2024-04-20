import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from rich.progress import Progress
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define dataset
class MoodDataset(Dataset):
    def __init__(self, features_1, features_2, labels_1, labels_2):
        self.features_1 = features_1
        self.features_2 = features_2
        self.labels_1 = labels_1
        self.labels_2 = labels_2

    def __len__(self):
        return len(self.labels_1)

    def __getitem__(self, idx):
        # Convert features and labels to tensors
        features_1 = torch.from_numpy(self.features_1[idx]).float()
        features_2 = torch.from_numpy(self.features_2[idx]).float()
        labels_1 = torch.from_numpy(np.array(self.labels_1[idx])).float()
        labels_2 = torch.from_numpy(np.array(self.labels_2[idx])).float()
        return features_1, features_2, labels_1, labels_2

# Define model
class TransformerModel(nn.Module):
    def __init__(self, task, d_model):
        super(TransformerModel, self).__init__()
        self.task = task
        self.transformer = nn.Transformer(d_model=d_model, nhead=7, batch_first=True, dropout=0.1)
        if self.task == 'regression':
            self.fc = nn.Linear(d_model, 1)
        elif self.task == 'classification':
            self.fc = nn.Linear(d_model, 3)


    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x)
        if self.task == 'regression':
            return x
        elif self.task == 'classification':
            return F.softmax(x, dim=-1)

class Data_loader:
    def __init__(self, df):
        self.df = df
        self.train_loader = None
        self.test_loader = None
        
    def data_preparation(self, batch_size=32):
        # Group by id
        grouped = self.df.groupby('id')

        features_1, features_2, labels_1, labels_2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for _, group in grouped:
            feature_1 = group.drop(['id', 'time', 'mood_type'], axis=1)
            feature_2 = group.drop(['id', 'time', 'mood_type'], axis=1)
            label_1 = group['mood']
            label_2 = group['mood_type']   

            feature_1 = feature_1.iloc[:-1].reset_index(drop=True)
            feature_2 = feature_2.iloc[1:].reset_index(drop=True)
            label_1 = label_1.iloc[1:].reset_index(drop=True)
            label_2 = label_2.iloc[1:].reset_index(drop=True)
            
            features_1 = pd.concat([features_1, feature_1])
            features_2 = pd.concat([features_2, feature_2])
            labels_1 = pd.concat([labels_1, label_1])
            labels_2 = pd.concat([labels_2, label_2])

        # Split into training and test sets
        train_features_1, test_features_1, train_features_2, test_features_2, train_labels_1, test_labels_1, train_labels_2, test_labels_2 = train_test_split(features_1.values, features_2.values, labels_1.values, labels_2.values, test_size=0.2, shuffle=False)
        train_features_1, test_features_1, train_features_2, test_features_2, train_labels_1, test_labels_1, train_labels_2, test_labels_2 = train_features_1.astype(np.float32), test_features_1.astype(np.float32), train_features_2.astype(np.float32), test_features_2.astype(np.float32), train_labels_1.astype(np.float32), test_labels_1.astype(np.float32), train_labels_2.astype(np.float32), test_labels_2.astype(np.float32)

        # Create data loaders
        train_dataset = MoodDataset(train_features_1, train_features_2, train_labels_1, train_labels_2)
        test_dataset = MoodDataset(test_features_1, test_features_2, test_labels_1, test_labels_2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_features = features_1.shape[1]
        
        return train_loader, test_loader, num_features

class Trainer:
    def __init__(self, task, model, device, criterion, lr, epoch):
        self.task = task
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
            print('\nStart training...')
            for i in range(self.epoch):
                total_loss = 0
                total_correct = 0
                total_count = 0
                for features_1, features_2, labels_1, labels_2 in train_loader:
                    features_1, features_2, labels_1, labels_2 = features_1.float().to(self.device), features_2.float().to(self.device), labels_1.float().squeeze().to(self.device), labels_2.float().squeeze().to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(features_1, features_2).squeeze()
                    if self.task == 'regression':
                        loss = self.criterion(outputs, labels_1)
                    elif self.task == 'classification':
                        loss = self.criterion(outputs, labels_2)
                    loss.backward()
                    self.optimizer.step()
                    # Calculate total loss and accuracy
                    total_loss += loss.item()
                    # outputs和labels为list，若outputs和labels中的一项的差值小于0.1，则认为这一项预测正确
                    total_correct += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels_1[i]) < 0.1])
                    total_count += labels_1.size(0)
                    
                # Print loss and accuracy for each epoch
                print(f'Epoch {i+1}: Loss = {total_loss/total_count}, Accuracy = {total_correct/total_count}')
                progress.update(task, advance=1)
        return self.model
                
    def save_model(self, path):
        # Save model
        torch.save(self.model.state_dict(), path)

class Tester:
    def __init__(self, task, model, device):
        self.task = task
        self.model = model
        self.device = device
        
    def test(self, test_loader):
        # Test model
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        print('\nStart testing...')
        with torch.no_grad():
            for i, (features_1, features_2, labels_1, labels_2) in enumerate(test_loader):
                features_1, features_2, labels_1, labels_2 = features_1.float().to(self.device), features_2.float().to(self.device), labels_1.float().to(self.device), labels_2.float().to(self.device)
                outputs = self.model(features_1, features_2).squeeze()
                total += labels_1.size(0)
                correct += sum([1 for i in range(len(outputs)) if abs(outputs[i] - labels_1[i]) < 0.1])
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels_1.cpu().numpy())
        
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        print('Accuracy: %d %%' % (100 * correct / total), 'MAE:', mae, 'MSE:', mse)
            
    def load_model(self, path):
        # Load model
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()

def Run_task(task, lr=1e-5, epoch=500, batch_size=128):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    if task == 'regression':
        criterion = nn.MSELoss()
    elif task == 'classification':
        criterion = nn.CrossEntropyLoss()
    
    # Load data
    df = pd.read_csv('Assignment1/time_resampling/featured_time_resamping_sparse_matrix_data.csv')
    data_loader = Data_loader(df)
    train_loader, test_loader, num_features = data_loader.data_preparation(batch_size)
    
    # Define model
    model = TransformerModel(task=task, d_model=num_features)
    
    trainer = Trainer(task, model, device, criterion, lr, epoch)
    model = trainer.train(train_loader)
    trainer.save_model('Assignment1/pytorch_model_'+str(task)+'.bin')
    
    # Test model
    tester = Tester(task, model, device)
    tester.load_model('Assignment1/pytorch_model_'+str(task)+'.bin')
    tester.test(test_loader)

if __name__ == '__main__':
    Run_task('regression')
    Run_task('classification')
