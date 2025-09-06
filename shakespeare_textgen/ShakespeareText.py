import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Data Loading and Preprocessing
with open('shakespeare-2.txt', 'r', encoding='utf-8') as f:
    words = f.read().split()

distinct_words = sorted(list(set(words)))
if '<PAD>' not in distinct_words:
    distinct_words = ['<PAD>'] + distinct_words

word_to_idx = {word: i for i, word in enumerate(distinct_words)}
idx_to_word = {i: word for i, word in enumerate(distinct_words)}

N_seq = 50
N_words = len(words)
N_vocab = len(distinct_words)

x_train, y_train = [], []
for i in range(N_words - N_seq):
    x_train.append([word_to_idx[w] for w in words[i:i+N_seq]])
    y_train.append(word_to_idx[words[i+N_seq]])

x_train = np.array(x_train, dtype=np.int64)
y_train = np.array(y_train, dtype=np.int64)

#Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

embedding_dim = 128
hidden_size = 512
model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab)

#Training Setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
PATH_SAVE = "shakespearean_generator_2.pth"

def save_checkpoint(model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH_SAVE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)


train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training Loop
num_epochs = 30
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_checkpoint(model, optimizer, epoch, best_loss)

#Text Generation
def generate(seed_words, N_words):
    model.eval()
    try:
        x0 = [word_to_idx[w] for w in seed_words]
    except KeyError as e:
        raise KeyError(f"Seed contains unknown word: {e}")

    if len(x0) < N_seq:
        x0 = [word_to_idx['<PAD>']] * (N_seq - len(x0)) + x0
    elif len(x0) > N_seq:
        x0 = x0[-N_seq:]

    generated_indices = x0.copy()

    for _ in range(N_words):
        x_tensor = torch.tensor([x0], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x_tensor), dim=1).cpu().numpy().ravel()
        idx = np.random.choice(N_vocab, p=probs)
        generated_indices.append(idx)
        x0 = x0[1:] + [idx]

    return generated_indices

#Seed
initial_seed = "your awesome character is very powerful today".lower()
seed_words = initial_seed.split()

invalid_words = set(seed_words) - set(word_to_idx.keys())
if invalid_words:
    raise SyntaxError(f"Invalid words: {invalid_words}")

if len(seed_words) > N_seq:
    seed_words = seed_words[-N_seq:]

N_pad = max(N_seq - len(seed_words), 0)
seed_words = ['<PAD>'] * N_pad + seed_words

print("The seed words are:", seed_words)

#Generate
generated_indices = generate(seed_words, 500)[N_pad:]
generated_sentence = ' '.join(
    idx_to_word[i] for i in generated_indices if idx_to_word[i] != '<UNK>'
)
print(generated_sentence)

#Save Model
torch.save(model.state_dict(), 'shakespeare_final.pth')
reloaded_model = LSTMModel(N_vocab, embedding_dim, hidden_size, N_vocab, num_layers=3)
reloaded_model.load_state_dict(torch.load('shakespeare_final.pth', map_location=device))
reloaded_model.to(device).eval()
