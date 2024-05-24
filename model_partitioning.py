import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import os
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Custom Dataset for IMDb Data (without torchtext)
class IMDBDataset(Dataset):
    def __init__(self, data_dir, split, word_to_index, vectors, max_len=256):
        self.data_dir = data_dir
        self.split = split
        self.data, self.labels = self.load_data()
        self.word_to_index = word_to_index  # Store the word_to_index mapping
        self.vectors = vectors            # Store the embedding vectors
        self.max_len = max_len

    def create_vocab_glove_map(self):
        vocab_glove_map = {}
        for word, idx in self.vocab.items():
            if word in self.glove.stoi:
                vocab_glove_map[idx] = self.glove.stoi[word]
            else:
                vocab_glove_map[idx] = 0  # Index 0 for unknown words
        return vocab_glove_map

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
        
        # Use word_to_index directly to get token ids
        token_ids = [self.word_to_index.get(token, self.word_to_index["<unk>"]) 
                     for token in text.split()][:self.max_len] 
        
        # Padding (pad with zeros)
        padding_length = self.max_len - len(token_ids)
        token_ids += [0] * padding_length
        
        return torch.tensor(token_ids), torch.tensor(label)

# Model Definition
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, output_dim, word_to_index, vectors, device):
        super().__init__()
        self.device = device  # Store device information
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=False).to(device)  # Load GloVe embeddings
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True).to(device)
        self.fc1 = nn.Linear(hidden_dim, 128).to(device)  # Additional layer for flexibility
        self.fc2 = nn.Linear(128, output_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        out = torch.relu(self.fc1(lstm_out))
        out = self.fc2(out)
        return out

# DataLoader setup (collate function is optional if you're okay with variable length sequences)
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  
    return padded_sequences, torch.tensor(labels)

def load_glove_vectors(glove_file_path):
    """Loads GloVe embeddings from a text file."""
    word_to_index = {}
    vectors = []

    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_to_index[word] = index
            vectors.append(vector)

    embedding_dim = len(vectors[0])

    # Add embeddings for <unk> and <pad> tokens
    word_to_index["<unk>"] = index + 1
    vectors.append(np.random.normal(scale=0.6, size=(embedding_dim,)))

    word_to_index["<pad>"] = index + 2
    vectors.append(np.zeros((embedding_dim,)))

    vectors = np.asarray(vectors, dtype='float32')

    return word_to_index, torch.tensor(vectors, dtype=torch.float32)

word_to_index, vectors = load_glove_vectors("glove.6B/glove.6B.100d.txt")

# Hyperparameters & Data Loading
batch_size = 64
num_epochs = 5  # Adjust as needed
learning_rate = 0.001

data_dir = "aclImdb"
dataset = IMDBDataset(data_dir, "train", word_to_index=word_to_index, vectors=vectors)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Model Initialization & Training
vocab_size = len(word_to_index) 
embed_dim = 100
hidden_dim = 256
num_layers = 5
output_dim = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, num_layers, output_dim, word_to_index=word_to_index, 
                            vectors=vectors, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
    # ... (Validation or other logging can be added here)

# Splitting the Model
model_part1 = nn.Sequential(model.embedding, model.lstm)
remaining_layers = list(model.children())[2:]  # Get layers after embedding and LSTM
model_part2 = nn.Sequential(*remaining_layers)

# Saving Models
torch.save(model_part1.state_dict(), "model_part1.pth")
torch.save(model_part2.state_dict(), "model_part2.pth")

# --- Model loading and evaluation ---
# Assuming model_part1 and model_part2 are already loaded:
model_part1.load_state_dict(torch.load("model_part1.pth"))
model_part2.load_state_dict(torch.load("model_part2.pth"))
model_part1.eval()
model_part2.eval()

with torch.no_grad():
    test_loss = 0
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device) 
        # Extract only the output tensor (discard hidden and cell states)
        output_part1, _ = model_part1(inputs)  
        # Take the last hidden state for classification
        output_part1 = output_part1[:,-1,:]  
        outputs = model_part2(output_part1)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        test_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs)).squeeze(1)
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")



def predict_sentiment(model_part1, model_part2, texts, word_to_index, max_len=256):   
    # Tokenize input texts
    tokenized_texts = []
    for text in texts:
        token_ids = [word_to_index.get(token, word_to_index["<unk>"]) for token in text.split()][:max_len]
        padding_length = max_len - len(token_ids)
        token_ids += [0] * padding_length
        tokenized_texts.append(torch.tensor(token_ids))
    
    # Create a DataLoader (single batch for efficiency)
    test_dataloader = DataLoader(tokenized_texts, batch_size=len(tokenized_texts))

    with torch.no_grad():
        for inputs in test_dataloader:  # Single batch iteration
            inputs = inputs.to(device)
            output_part1, _ = model_part1(inputs)
            output_part1 = output_part1[:,-1,:]
            outputs = model_part2(output_part1)
            predictions = torch.round(torch.sigmoid(outputs)).squeeze().tolist()

    return predictions

test_texts = [
    "This movie is a masterpiece! The acting was superb.",
    "I really enjoyed this film. Truly captivating.",
    "This was the worst movie I've ever seen. It was boring and predictable.",
    "The acting was terrible. The special effects were laughable."
]

predictions = predict_sentiment(model_part1, model_part2, test_texts, word_to_index)

# Print results with original texts
for text, sentiment in zip(test_texts, predictions):
    print(f'Text: "{text}"\nSentiment: {"Positive" if sentiment == 1 else "Negative"}\n')

