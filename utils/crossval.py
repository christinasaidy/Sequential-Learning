from sklearn.model_selection import KFold
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 

class CrossValidation:
    def __init__(self, dataset, n_splits=5):
        self.dataset = dataset
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True)

    def run(self, model_class, epochs, lr, device, vocab, embed_dim, hidden_dim, embedding_layer=None):
        accuracies = []
        for train_index, val_index in self.kf.split(self.dataset):
            train_subset = torch.utils.data.Subset(self.dataset, train_index.tolist())
            val_subset = torch.utils.data.Subset(self.dataset, val_index.tolist())

            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            if model_class.__name__ == "RNN_Classifier":
                model = model_class(embedding_layer, hidden_dim, 4).to(device)
            else:
                model = model_class(len(vocab) + 2, embed_dim, hidden_dim, 4).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.CrossEntropyLoss().to(device)

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                total_correct = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == labels).sum().item()

                avg_loss = total_loss / len(train_subset)
                accuracy = total_correct / len(train_subset)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

            model.eval()
            total_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == labels).sum().item()

            accuracy = total_correct / len(val_subset)
            accuracies.append(accuracy)
            print(f"Validation Accuracy: {accuracy:.4f}")

        return np.mean(accuracies), np.std(accuracies)
