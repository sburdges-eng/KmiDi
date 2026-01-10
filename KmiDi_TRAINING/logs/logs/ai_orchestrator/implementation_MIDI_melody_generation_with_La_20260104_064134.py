"""
Generated Implementation: MIDI melody generation with Lakh MIDI dataset - best architectures and training strategies
Based on deep research conducted 2026-01-04T06:41:34.124616
"""

import torch
from torch import nn
import torch.nn.functional as F

class MelodyTransformer(nn.Module):
    """
    A Transformer-based model for melody generation.

    This model replaces the traditional LSTM architecture with a Transformer
    to better capture long-range dependencies and complex patterns in music sequences.

    Attributes:
        vocab_size (int): The size of the vocabulary (number of unique notes).
        d_model (int): The dimension of the embedding space.
        nhead (int): The number of attention heads.
        num_layers (int): The number of Transformer layers.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super(MelodyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)  # Regularization to prevent overfitting

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Args:
            src (Tensor): Source sequence tensor (input melody).
            tgt (Tensor): Target sequence tensor (output melody).

        Returns:
            Tensor: The output logits for each step in the sequence.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(self.dropout(output))
        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding module to inject information about the position of each note.

    This helps the Transformer model to capture the order of notes in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: The input tensor with positional encoding added.
        """
        return x + self.encoding[:, :x.size(1), :]

# Example usage
if __name__ == "__main__":
    # Define model parameters
    vocab_size = 128  # Example vocabulary size for MIDI notes
    model = MelodyTransformer(vocab_size)

    # Example input sequences (batch_size, seq_len)
    src = torch.randint(0, vocab_size, (32, 100))  # Random source sequence
    tgt = torch.randint(0, vocab_size, (32, 100))  # Random target sequence

    # Forward pass
    output = model(src, tgt)
    print(output.shape)  # Expected output shape: (seq_len, batch_size, vocab_size)
