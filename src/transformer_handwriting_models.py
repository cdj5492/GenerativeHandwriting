
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import bernoulli, uniform
from utils.model_utils import stable_softmax


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
        y_hat.device
    )
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample


def sample_batch_from_out_dist(y_hat, bias):
    batch_size = y_hat.shape[0]
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=1)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=1)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1).squeeze()

    mu_k = y_hat.new_zeros((y_hat.shape[0], 2))

    mu_k[:, 0] = mu_1[torch.arange(batch_size), K]
    mu_k[:, 1] = mu_2[torch.arange(batch_size), K]
    cov = y_hat.new_zeros(y_hat.shape[0], 2, 2)
    cov[:, 0, 0] = std_1[torch.arange(batch_size), K].pow(2)
    cov[:, 1, 1] = std_2[torch.arange(batch_size), K].pow(2)
    cov[:, 0, 1], cov[:, 1, 0] = (
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
        correlations[torch.arange(batch_size), K]
        * std_1[torch.arange(batch_size), K]
        * std_2[torch.arange(batch_size), K],
    )

    X = torch.normal(
        mean=torch.zeros(batch_size, 2, 1), std=torch.ones(batch_size, 2, 1)
    ).to(y_hat.device)
    Z = mu_k + torch.matmul(cov, X).squeeze()

    sample = y_hat.new_zeros(batch_size, 1, 3)
    sample[:, 0, 0:1] = eos_sample
    sample[:, 0, 1:] = Z.squeeze()
    return sample


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional embeddings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to a batch of sequences (N, T, D)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Causal mask to prevent attention to future positions."""
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask.triu_(1)
    return mask


# --------------------------------------------------------------------------- #
# 1) Unconditional prediction model
# --------------------------------------------------------------------------- #

class HandWritingPredictionNet(nn.Module):
    """Transformer encoder that predicts next-point Mixture-Density params."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        input_size: int = 3,
        output_size: int = 121,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (N, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        inputs: FloatTensor (batch, seq_len, 3) - (eos, dx, dy)
        src_key_padding_mask: BoolTensor (batch, seq_len) - True for padding.
        """
        x = self.input_proj(inputs)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        y_hat = self.output_head(x)
        return y_hat

    @torch.no_grad()
    def generate(
        self,
        prime: torch.Tensor, # Note: prime handling is still a TODO
        seq_len: int,
        bias: float,
        # Add device parameter for creating the initial tensor
        device: Optional[torch.device] = None,
        batch_size: int = 1 # Assuming batch size 1 for generation if not specified
    ) -> torch.Tensor:
        """Autoregressively sample *seq_len* points after *prime*."""
        
        if device is None:
             # Try to infer device from model parameters if not provided
             device = next(self.parameters()).device

        if prime is not None and prime is not False: # Check if prime is a tensor
             generated = [prime.to(device)]
             current_batch_size = prime.shape[0]
             # Ensure prime has the correct shape (B, T, 3)
             if prime.dim() != 3 or prime.shape[-1] != 3:
                  raise ValueError("Prime tensor must have shape (batch_size, sequence_length, 3)")

        else:
             # Start with a beginning-of-sequence token (e.g., zeros)
             # Shape: (batch_size, 1, 3) -> (eos=0, dx=0, dy=0)
             initial_point = torch.zeros((batch_size, 1, 3), device=device)
             generated = [initial_point]
             current_batch_size = batch_size


        # Autoregressive loop
        for _ in range(seq_len):
            # Concatenate the points generated so far to form the context
            context = torch.cat(generated, dim=1) # Now 'generated' is not empty
            
            # Get model prediction for the next point based on context
            y_hat = self.forward(context)
            
            # Extract the prediction for the *last* time step
            y_last = y_hat[:, -1, :] # Shape (batch_size, output_size)

            # Sample the next point based on the prediction distribution
            # Ensure sample_batch_from_out_dist can handle batch input
            # Assuming sample_batch_from_out_dist returns shape (batch_size, 1, 3)
            z = sample_batch_from_out_dist(y_last, bias)
            
            # Append the newly generated point
            generated.append(z)

        # Concatenate all generated points (excluding the initial if it was just a placeholder)
        # If we started with a placeholder zero, we might want to skip it in the final output,
        # depending on whether it represents a real starting point or just context.
        # If the initial_point was just for context, return torch.cat(generated[1:], dim=1)
        # Otherwise, if it's part of the sequence:
        return torch.cat(generated, dim=1) # Shape: (batch_size, initial_len + seq_len, 3)


# --------------------------------------------------------------------------- #
# 2) Text-conditioned synthesis model (encoder-decoder Transformer)
# --------------------------------------------------------------------------- #

class HandWritingSynthesisNet(nn.Module):
    """Cross-attention Transformer for text-conditioned handwriting."""

    def __init__(
        self,
        vocab_size: int = 77,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        output_size: int = 121,
    ):
        super().__init__()
        self.d_model = d_model

        # Text encoder
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos_enc = PositionalEncoding(d_model, dropout) 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Stroke decoder
        self.stroke_proj = nn.Linear(3, d_model)
        self.stroke_pos_enc = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_head = nn.Linear(d_model, output_size)

    def forward(
        self,
        stroke_input: torch.Tensor,
        text: torch.Tensor,
        stroke_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        stroke_input: (batch, T_stroke, 3)
        text: (batch, T_text)
        """
        # Encode text
        text_emb = self.text_pos_enc(self.text_embed(text) * math.sqrt(self.d_model))
        memory = self.encoder(text_emb, src_key_padding_mask=text_padding_mask)

        # Prepare stroke target
        tgt = self.stroke_pos_enc(self.stroke_proj(stroke_input))
        tgt_mask = _generate_square_subsequent_mask(tgt.size(1), tgt.device)

        dec_out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=stroke_padding_mask,
            memory_key_padding_mask=text_padding_mask,
        )
        y_hat = self.output_head(dec_out)
        return y_hat

    @torch.no_grad()
    def generate(
        self,
        prime: torch.Tensor,
        text: torch.Tensor,
        seq_len: int,
        bias: float,
    ) -> torch.Tensor:
        """Generate handwriting sequence conditioned on *text*."""
        text_pad_mask = None  # assume trimmed
        memory = self.encoder(
            self.text_pos_enc(self.text_embed(text) * math.sqrt(self.d_model)),
            src_key_padding_mask=text_pad_mask,
        )

        generated = [prime]
        for _ in range(seq_len):
            tgt = torch.cat(generated, dim=1)
            tgt_emb = self.stroke_pos_enc(self.stroke_proj(tgt))
            tgt_mask = _generate_square_subsequent_mask(tgt_emb.size(1), tgt.device)
            dec_out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=text_pad_mask,
            )
            y_last = self.output_head(dec_out)[:, -1, :]
            z = sample_from_out_dist(y_last.squeeze(), bias)
            generated.append(z)
        return torch.cat(generated, dim=1)