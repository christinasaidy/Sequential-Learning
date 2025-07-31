from torch.utils.data import Dataset 
import torch 

class AGNewsDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df['encoded'].tolist(), dtype=torch.long)
        self.y = torch.tensor(df['label'].tolist(), dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]