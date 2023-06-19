import torch
from torch import nn

class GPT2(nn.Module):
    """Implements a simplified version of the GPT-2 model using PyTorch's built-
    in Transformer module.

    This module uses TransformerDecoder for the core GPT-2 architecture and
    learns embeddings for both the words and their positions.

    Attributes:
        embed_size:  The size of the word embeddings.
        word_embedding:  The learnable word embeddings.
        position_embedding:  The learnable position embeddings.
        decoder_layer:  Single Transformer Decoder layer used in the Transformer
        transformer_decoder:  The multi-layer Transformer Decoder.
        fc_out:  The final output linear layer.
        dropout:  Dropout layer used for regularization.
        device:  The device type on which to do computations.
    """
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 forward_expansion,
                 dropout,
                 max_length):
        """Initializes the GPT2 model with given parameters.
        
        Args:
            vocab_size:  The size of the vocabulary.
            embed_size:  The size of the word embeddings.
            num_layers:  The number of Transformer Decoder layers.
            num_heads:  The number of attention heads in the Transformer Decoder.
            forward_expansion:  The expansion factor for the feedforward layer.
            dropout:  The dropout rate used in dropout layers for regularization.
            max_length:  The maximum length of the input sequence.
        """
        super().__init__()
        self.embed_size = embed_size
        # Learnable word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # Learnable position embeddings
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Construct the decoder layer to be used in the transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion*embed_size,
            dropout=dropout)
        # Construct the transformer decoder with the above decoder layer
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=num_layers)
        
        self.create_mask = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=forward_expansion*embed_size,
            dropout=dropout).generate_square_subsequent_mask
        
        # Final output linear layer
        self.fc_out = nn.Linear(embed_size, vocab_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Decide the computation device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device) # Moves the entire model to the device
    
    def forward(self, x, mask):
        """Defines the forward pass for the GPT-2 model.
        
        Args:
            x:  The input tensor of shape (batch_size, seq_length).
            mask: The attention mask for the transformer decoder.
            
        Returns:
            The output tensor after passing through the model."""
        # Get the batch size and sequence length from the input tensor
        x = x.to(self.device) # Ensure x is on the correct device
        # Check if a mask was provided
        if mask:
            mask = self.create_mask(x.size(1),self.device)  # Ensure mask is on the correct device
        else:
            mask = None

        N, seq_length = x.shape

        # Create position embeddings for the input sequence
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Apply dropout to the sum of the word and position embeddings
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Permute the dimensions to match the expected shape for the transformer
        out = out.permute(1, 0, 2)

        # Pass through the transformer decoder.  The decoder uses the embeddings
        # as both the input and the target sequence since its a decoder-only
        # language model.
        # Only pass the mask if it's not None
        out = self.transformer_decoder(out, out, mask) if mask is not None else self.transformer_decoder(out, out)

        # Permute the dimensions back to the original shape
        out = out.permute(1, 0, 2)

        # Pass through the final output linear layer to get hte logits.
        out = self.fc_out(out)

        return out