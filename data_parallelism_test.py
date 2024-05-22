import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # Import functional module
import nltk
from nltk.tokenize import sent_tokenize  # Import sentence tokenizer

nltk.download('punkt')

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

# Create the model instances FIRST
model1 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)
model2 = SimpleClassifier(len(dataset.vocab), 100, 128, 2).to(device)

# Load Saved Models
model1.load_state_dict(torch.load("data-parallelism-model1.pth"))
model2.load_state_dict(torch.load("data-parallelism-model2.pth"))

# Choose Model for Testing (e.g., model1)
test_model1 = model1
test_model1.eval()

test_model2 = model2
test_model2.eval()


def predict_sentiment_split_aggregate(text, models, vocab, device):
    sentences = sent_tokenize(text)
    model_outputs = []

    for i, sentence in enumerate(sentences):
        model = models[i % len(models)]
        tokenized_sentence = [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]
        processed_sentence = torch.tensor([tokenized_sentence], device=device)
        
        with torch.no_grad():
            output = model(processed_sentence)
            model_outputs.append(output)

    # Aggregate probabilities using a simple average (you can experiment with weighted average)
    probabilities = [F.softmax(output, dim=1) for output in model_outputs]
    confidences = [torch.max(F.softmax(output, dim=1), dim=1)[0] for output in model_outputs]
    weights = F.softmax(torch.tensor(confidences), dim=0)
    weighted_probabilities = sum(weight * prob for weight, prob in zip(weights, probabilities))
    predicted_label = torch.argmax(weighted_probabilities).item()

    return "Positive" if predicted_label == 1 else "Negative"


# Example Usage
test_texts = [
    "This movie is a masterpiece! The acting was superb.",
    "I really enjoyed this film. Truly captivating.",
    "This was the worst movie I've ever seen. It was boring and predictable.",
    "The acting was terrible. The special effects were laughable."
]

aggregated_sentiments = [predict_sentiment_split_aggregate(text, [test_model1, test_model2], dataset.vocab, device) 
                         for text in test_texts]

for text, sentiment in zip(test_texts, aggregated_sentiments):
    print(f"'{text}' - Aggregated Sentiment: {sentiment}")