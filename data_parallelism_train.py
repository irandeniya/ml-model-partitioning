import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# Model Definition
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text):
        embedded = self.embedding(text)
        x = self.fc1(embedded.mean(dim=1))  # Average embeddings for simplicity
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Custom Dataset for IMDb Data (without torchtext)
class IMDBDataset(Dataset):
    def __init__(self, data_dir, split, vocab=None, max_len=256):
        self.data_dir = data_dir
        self.split = split
        # Load data from files
        self.data, self.labels = self.load_data()

        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.max_len = max_len

    def load_data(self):
        data = []
        labels = []
        for label in ["pos", "neg"]:
            label_dir = os.path.join(self.data_dir, self.split, label)
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        return data, labels

    def build_vocab(self):
        all_text = " ".join(self.data)
        token_counts = Counter(all_text.split())  
        vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_counts.most_common())}  
        vocab["<unk>"] = 0
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in text.split()][:self.max_len]  
        return torch.tensor(token_ids), torch.tensor(label)

# DataLoader setup (collate function is optional if you're okay with variable length sequences)
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  
    return padded_sequences, torch.tensor(labels)

# Data Preparation
data_dir = "aclImdb"  # Update with your path
dataset = IMDBDataset(data_dir, "train")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Use only the first GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)
model2 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)

optimizer1 = optim.Adam(model1.parameters())
optimizer2 = optim.Adam(model2.parameters())
loss_function = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


# Training Loop for Each Model Separately
def train(dataloader, model, optimizer, loss_fn, device):
    model.train()
    for batch in dataloader:
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            test_loss += loss_fn(output, labels).item()  # Sum up batch loss

            _, predicted = torch.max(output, 1)  # Get predicted class labels
            total += labels.size(0)  # Total number of samples in the batch
            correct += (predicted == labels).sum().item()  # Count correct predictions

    test_loss /= len(dataloader)  # Average loss across all batches
    accuracy = 100.0 * correct / total

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return accuracy

num_epochs = 5
for epoch in range(num_epochs):
    train(train_loader, model1, optimizer1, loss_function, device)  # Train model 1
    print(f"Epoch {epoch + 1}, Model 1: Finished")
    test(test_loader, model1, loss_function, device)  # Test model 1
    
    train(train_loader, model2, optimizer2, loss_function, device)  # Train model 2
    print(f"Epoch {epoch + 1}, Model 2: Finished")
    test(test_loader, model2, loss_function, device)  # Test model 2


# Save Models
torch.save(model1.state_dict(), "data-parallelism-model1.pth")
torch.save(model2.state_dict(), "data-parallelism-model2.pth")

