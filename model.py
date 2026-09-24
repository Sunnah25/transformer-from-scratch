import torch
import torch.nn as nn

class Head(nn.Module):
    """Single attention head"""
    def __init__(self, head_size, embed_size):
        super().__init__()

        # These three lines create W_Q, W_K, W_V
        # Each transforms embed_size → head_size

        self.head_size = head_size
        self.Q = nn.Linear(embed_size, head_size)
        self.K = nn.Linear(embed_size, head_size)
        self.V = nn.Linear(embed_size, head_size)

    def forward(self, x):
        # x shape: (batch, sequence_length, embed_size)
        q = self.Q(x) #(batch, seq_len, head_size)
        k = self.K(x) #(batch, seq_len, head_size)
        v = self.V(x) #(batch, seq_len, head_size)


        #Attention scores : Q × K^T
        # This asks: how much each token attend to each other token?
        scores = q @ k.transpose(-2, -1)

        #Scale by sqrt(head_size) - prevents scores getting too large
        scores = scores/ (self.head_size ** 0.5)

        #Softmax - convert scores to probabilities
        weights = torch.softmax(scores, dim=-1) #(batch, seq_len, seq_len)

        # Weighted sum of values
        output = weights @ v

        return output






class MultiHead(nn.Module):
    """Multiple attention heads"""
    def __init__(self, num_heads, head_size, embed_size):
        super().__init__()

        #Create num_heads attention heads
        self.heads = nn.ModuleList([
            Head(head_size, embed_size)
            for _ in range(num_heads)
        ])
        # After concatenating all heads, project back to embed_size
        self.projection = nn.Linear(num_heads * head_size, embed_size)



    def forward(self, x):
        # step 1 - run each head
        head_outputs = [head(x) for head in self.heads]

        #step 2 - concatanate along last dimension
        x = torch.cat(head_outputs, dim=-1)

        # step 3 - project back to embed_size
        output = self.projection(x)

        return output




    

class FFN(nn.Module):
    """Feed Forward Network"""
    def __init__(self, embed_size):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, 4*embed_size)
        self.layer2 = nn.Linear(4*embed_size, embed_size)


    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x








class Block(nn.Module):
    """One transformer block"""
    def __init__(self, num_heads, head_size, embed_size):
        super().__init__()
        self.attention = MultiHead(num_heads, head_size, embed_size)
        self.ffn = FFN(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x





    

class Transformer(nn.Module):
    """Full model"""
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, seq_len, head_size):
        super().__init__()

        # 1. Token embedding - converts token ids to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # 2. Positional embedding - learns position signals
        self.pos_embedding = nn.Embedding(seq_len, embed_size)

        # 3. Stack of blocks - num_layers Block objects
        self.stack_blocks = nn.Sequential(*[Block(num_heads, head_size, embed_size) for _ in range(num_layers)])

        # 4. Final LayerNorm
        self.final_layer = nn.LayerNorm(embed_size)

        # 5. Output head - projects to vocubulary size
        self.project = nn.Linear(embed_size, vocab_size)


    def forward(self, x):
        batch, seq_len = x.shape

        # Step 1 - token embeddings
        tok_emb = self.token_embedding(x) #(batch, seq_len, embed_size)

        # Step 2 - positonal embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)

        # Step 3 - add together (broadcast automatically)
        x = tok_emb + pos_emb

        # Step 4 - pass through blocks
        x = self.stack_blocks(x)

        # Step 5 - apply final norm
        x = self.final_layer(x)

        # Step 6 - apply output head
        x = self.project(x)

        return x

"""
# test

#Hyperparameters
vocab_size = 65     # num of unique characters
embed_size = 32     # size of embeddings
num_heads = 4       # number of attention heads
head_size = 8       # embed_size // num_heads
num_layers = 3      # number of transformer blocks
seq_len = 10        # sequence length
batch = 2           # batch size


#Create model
model = Transformer(vocab_size, embed_size, num_heads, num_layers, seq_len, head_size)

#Fake input - batch of token ids
x = torch.randint(0, vocab_size, (batch, seq_len))
print(f"Input shape: {x.shape}")


#Forward pass
output = model(x)
print(f"Output shape: {output.shape}")

#Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


"""