import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self, embedding_dim, heads, qkv_bias=False):
    super(SelfAttention, self).__init__()
    
    self.embedding_dim = embedding_dim
    self.heads = heads
    self.head_dim = embedding_dim // heads
    # feed head_dim number of parts into each self-attention head
    
    assert (self.head_dim * heads == embedding_dim), "Embedding dimension must be perfectly divisible by number of heads"

    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
    self.values = nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
    
    self.fc_out = nn.Linear(embedding_dim, embedding_dim) # after concatenating output from all heads, input_dim will be head_dim*heads
    
  def forward(self, key, query, value, mask=None):
    N = query.shape[0] # N is batch size (number of examples that are fed into the encoder at a time)
    key_len, query_len, value_len = key.shape[1], query.shape[1], value.shape[1]
    
    # split embedding into self.heads pieces (embedding_dim==heads*head_dim)
    query = query.reshape(N, query_len, self.heads, self.head_dim)
    key = key.reshape(N, key_len, self.heads, self.head_dim)
    value = value.reshape(N, value_len, self.heads, self.head_dim)
    
    query = self.queries(query)
    key = self.keys(key)
    value = self.values(value)
    
    # multiply query and key -> einsum is similar to matmul, but is much easier
    energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
    # query shape and key shape are above
    # energy shape is : (N, heads, query_len, key_len)
    # normallly we would use torch.bmm
    
    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))
      
    attention = torch.softmax(energy / (self.embedding_dim ** (1/2)), dim=3)
    
    attention = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
      N, query_len, self.embedding_dim
    )
    # attention shape is : (N, heads, query_len, key_len)
    # values shape is : (N, value_len, heads, head_dim)
    # (N, query_len, heads, head_dim) -> this is our desired output
    # after einsum, flatten the last two dimensions (embedding_dim==heads*head_dim)
    
    out = self.fc_out(attention)
    
    return out
  
class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    
    self.mha = SelfAttention(embedding_dim, heads)
    self.norm1 = nn.LayerNorm(embedding_dim)
    self.norm2 = nn.LayerNorm(embedding_dim)
    
    # feed forward block doesn't change any dimensions, it just does some learning (computations) by using forward_expansion
    self.ff = nn.Sequential(
      nn.Linear(embedding_dim, forward_expansion*embedding_dim),
      nn.ReLU(),
      nn.Linear(forward_expansion*embedding_dim, embedding_dim)
    )
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, key, query, value, mask):
    mha = self.mha(key, query, value, mask)
    x = self.dropout(self.norm1(mha + query))
    
    forward = self.ff(x)
    out = self.dropout(self.norm2(forward + x)) # (forward + x) is the skip connection
    
    return out
    
class Encoder(nn.Module):
  def __init__(self, input_vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_len): # max_len is related to positional embedding, it is the maximum sentence length
    super(Encoder, self).__init__()
    
    self.embedding_dim = embedding_dim
    self.device = device
    self.word_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=embedding_dim)
    self.positional_encoding = nn.Embedding(num_embeddings=max_len, embedding_dim=embedding_dim)
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embedding_dim,
          heads,
          dropout,
          forward_expansion
        )
        for _ in range(num_layers)
      ]
    )
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x, mask):
    N, seq_len = x.shape # x is a batch of input sequences (N is the batch size)
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(device=self.device)
    
    encoder_input = self.dropout(self.word_embedding(x) + self.positional_encoding(positions))
    out = None
    # this is the input to the encoder block, which includes the word embedding and the positional encoding
    
    for layer in self.layers:
      out = layer(encoder_input, encoder_input, encoder_input, mask) # right now we are dealing with SELF-ATTENTION, hence the key, query and value will all be the same
      
    return out
  
class DecoderBlock(nn.Module):
  def __init__(self, embedding_dim, heads, forward_expansion, dropout, device):
    super(DecoderBlock, self).__init__()
    self.mha = SelfAttention(embedding_dim, heads)
    self.embedding_dim = embedding_dim
    self.norm = nn.LayerNorm(embedding_dim)
    self.transformer_block = TransformerBlock(
      embedding_dim=embedding_dim,
      heads=heads,
      dropout=dropout,
      forward_expansion=forward_expansion
    )
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x, key, value, source_mask, target_mask):
    attention = self.mha(x, x, x, target_mask)
    query = self.dropout(self.norm(attention + x))
    
    out = self.transformer_block(key, query, value, source_mask)
    return out
  
class Decoder(nn.Module):
  def __init__(self, output_vocab_size, embedding_dim, num_layers, heads, forward_expansion, dropout, device, max_len):
    super(Decoder, self).__init__()
    self.device = device
    self.word_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=embedding_dim)
    self.positional_encoding = nn.Embedding(num_embeddings=max_len, embedding_dim=embedding_dim)
    
    self.layers = nn.ModuleList(
      [
        DecoderBlock(embedding_dim, heads, forward_expansion, dropout, device)
        for _ in range(num_layers)
      ]
    )
    
    self.fc_out = nn.Linear(embedding_dim, output_vocab_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x, encoder_out, source_mask, target_mask):
    N, seq_len = x.shape
    positions = torch.arange(0, seq_len).expand(N, seq_len).to(device=self.device)
    x = self.dropout((self.word_embedding(x) + self.positional_encoding(positions)))
    
    for layer in self.layers:
      x = layer(x, encoder_out, encoder_out, source_mask, target_mask) # the two outputs from encoder is fed into the decoder, as shown in the image in the paper
      
    out = self.fc_out(x)
    
    return out
    
class Transformer(nn.Module):
  def __init__(self, input_vocab_size, output_vocab_size, input_padding_index, output_padding_index, device, embedding_dim=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, max_len=100):
    super(Transformer, self).__init__()
    
    self.encoder = Encoder(input_vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_len)
    self.decoder = Decoder(output_vocab_size, embedding_dim, num_layers, heads, forward_expansion, dropout, device, max_len)
    
    self.input_padding_index = input_padding_index
    self.output_padding_index = output_padding_index
    self.device = device
    
  def make_input_mask(self, input):
    input_mask = (input !=  self.input_padding_index).unsqueeze(1).unsqueeze(2)
    return input_mask.to(device=self.device)
  
  def make_output_mask(self, output):
    N, output_len = output.shape
    output_mask = torch.tril(torch.ones((output_len, output_len))).expand(
      N, 1, output_len, output_len
    )
    
    return output_mask.to(device=self.device)
  
  def forward(self, input, output):
    input_mask = self.make_input_mask(input)
    output_mask = self.make_output_mask(output)
    
    enc_out = self.encoder(input, input_mask)
    out = self.decoder(output, enc_out, input_mask, output_mask)
    
    return out
  
if __name__=="__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  input = torch.tensor([[7, 2, 4, 1, 8, 6, 5, 3, 0], [9, 2, 6, 4, 7, 1, 3, 2, 8]]).to(device=device)
  output = torch.tensor([[4, 8, 2, 6, 1, 7, 3], [3, 7, 1, 5, 2, 6, 4]]).to(device=device)
  
  input_padding_index = 0
  output_padding_index = 0
  input_vocab_size = 10
  output_vocab_size = 10
  
  model = Transformer(input_vocab_size, output_vocab_size, input_padding_index, output_padding_index, device).to(device)
  
  out = model(input, output[:, :-1]) # shifting target by one, we don't want it to have the EOS token
  
  print(out)
  print(out.shape)
    
    