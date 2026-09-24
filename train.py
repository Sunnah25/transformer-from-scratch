import torch
from model import Transformer


# Hyperparameters
vocab_size = 65
embed_size = 64
num_heads = 4
head_size = embed_size // num_heads  # 16
num_layers = 3
seq_len = 32
batch_size = 16
learning_rate = 1e-3
max_steps = 1000


with open("input.txt", "r") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_int = {}
int_to_char = {}


for i, char in enumerate(chars):
    char_to_int[char] = i
    int_to_char[i] = char


def encode(input):
    output_arr = []

    for ch in input:
        output_arr.append(char_to_int[ch])

    return output_arr


def decode(the_num_arr):
    str_arr = []
    result = ""

    for element in the_num_arr:
        str_arr.append(int_to_char[element])
    result = "".join(str_arr)

    return result







# Encode entire text
data = torch.tensor(encode(text), dtype=torch.long)
#print(f"Data shape: {data.shape}")
#print(f"First 20 tokens: {data[:20]}")



# Train/validation split
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
#print(f"Train size: {len(train_data)}")
#print(f"Val size: {len(val_data)}")


seq_len = 32
batch_size = 16

def get_batch(data):
    # Pick batch_size random starting positions
    positions = torch.randint(0, len(data) - seq_len, (batch_size,))
    
    # Grab seq_len characters from each position → input
    x = torch.stack([data[i:i+seq_len] for i in positions])
    
    # Grab seq_len characters shifted by 1 → targets
    y = torch.stack([data[i+1:i+seq_len+1] for i in positions])
    
    return x, y

# Test it
#x, y = get_batch(train_data)
#print(f"x shape: {x.shape}")
#print(f"y shape: {y.shape}")
#print(f"x sample: {decode(x[0].tolist())}")
#print(f"y sample: {decode(y[0].tolist())}")



model = Transformer(vocab_size, embed_size, num_heads, num_layers, seq_len, head_size)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


for step in range(max_steps):
    # 1. Get a batch
    x, y = get_batch(train_data)
    
    # 2. Forward pass — get predictions
    output = model(x)
    
    # 3. Calculate loss
    # output shape: (batch, seq_len, vocab_size) — need to reshape for loss
    # y shape: (batch, seq_len) — need to reshape too
    loss = torch.nn.functional.cross_entropy(
        output.view(batch_size * seq_len, vocab_size),
        y.view(batch_size * seq_len)
    )
    
    # 4. Zero gradients
    optimiser.zero_grad()
    
    # 5. Backward pass
    loss.backward()
    
    # 6. Update weights
    optimiser.step()
    
    # 7. Print loss every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")




def generate(model, start_text, max_new_tokens=200):
    # Encode starting text
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    
    generated = list(encode(start_text))
    
    for _ in range(max_new_tokens):
        # Crop context to seq_len
        context_crop = context[:, -seq_len:]
        
        # Get predictions
        output = model(context_crop)
        
        # Focus on last token's predictions
        last_token_logits = output[0, -1, :]
        
        # Convert to probabilities
        probs = torch.softmax(last_token_logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Add to generated sequence
        generated.append(next_token)
        context = torch.cat([context, torch.tensor([[next_token]])], dim=1)
    
    return decode(generated)

# Generate text
print("\n--- Generated Text ---")
print(generate(model, "To be or not to be"))