import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # Import functional module

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


# Use only the first GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Preparation
data_dir = "aclImdb"  # Update with your path
dataset = IMDBDataset(data_dir, "train")

model1 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)
model2 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)

model1.load_state_dict(torch.load("model-parallelism-model1.pth"))
model2.load_state_dict(torch.load("model-parallelism-model2.pth"))

test_model1 = model1
test_model1.eval()

test_model2 = model2
test_model2.eval()

def aggregate_sentiment_with_probabilities(texts, models, vocab, device):
    tokenized_texts = [[vocab.get(token, vocab["<unk>"]) for token in text.split()] for text in texts]
    max_len = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_texts]

    processed_texts = torch.tensor(padded_texts, device=device)

    with torch.no_grad():
        outputs = [model(processed_texts) for model in models]
        probabilities = [F.softmax(output, dim=-1) for output in outputs]  # Probabilities for each model

    # Average probabilities across models
    avg_probabilities = torch.mean(torch.stack(probabilities), dim=0)
    predicted_labels = torch.argmax(avg_probabilities, dim=1)

    return ["Positive" if label.item() == 1 else "Negative" for label in predicted_labels]

# Example Usage (Batch Inference and Aggregation)
test_texts = [
    "This movie is a masterpiece!",
    "I really enjoyed this film.",
    "This was the worst movie I've ever seen.",
    "The acting was terrible."
]

aggregated_sentiments = aggregate_sentiment_with_probabilities(test_texts, [test_model1, test_model2], 
                                                               dataset.vocab, device)

for text, sentiment in zip(test_texts, aggregated_sentiments):
    print(f"'{text}' - Aggregated Sentiment: {sentiment}")


