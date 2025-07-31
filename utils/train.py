import torch

def train_model(model, train_loader, train_dataset, loss_fn, optimizer, device, epochs):
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
        avg_loss = total_loss / len(train_dataset)
        accuracy = total_correct / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")