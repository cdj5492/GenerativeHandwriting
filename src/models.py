import torch
import math
import torch.nn as nn
# Import Transformer classes
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import bernoulli, uniform
from utils.model_utils import stable_softmax
import numpy as np
import math
import torch.nn.functional as F


def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    # Ensure y_hat has the correct dimension before splitting
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0) # Add batch dim if missing
    elif y_hat.dim() == 0: # Handle scalar case if it somehow occurs
         y_hat = y_hat.unsqueeze(0).unsqueeze(0)

    # Adjust split dimension based on y_hat shape
    split_dim = 1 if y_hat.dim() > 1 else 0
    y = torch.split(y_hat, split_sizes, dim=split_dim)


    eos_prob = torch.sigmoid(y[0])
    # Adjust softmax dimension
    softmax_dim = 1 if y[1].dim() > 1 else 0
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=softmax_dim)

    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    # Ensure mixture_weights is 2D for multinomial
    if mixture_weights.dim() == 1:
        mixture_weights = mixture_weights.unsqueeze(0)
    K = torch.multinomial(mixture_weights, 1).squeeze() # Squeeze potential batch dim if size 1

    # Ensure K is treated as an index, handle scalar case
    k_index = K.item() if K.numel() == 1 else K

    # Initialize mu_k correctly
    mu_k = y_hat.new_zeros(2)

    # Index correctly, handling potential scalar outputs from split
    mu_k[0] = mu_1.squeeze()[k_index] if mu_1.numel() > 1 else mu_1.item()
    mu_k[1] = mu_2.squeeze()[k_index] if mu_2.numel() > 1 else mu_2.item()

    cov = y_hat.new_zeros(2, 2)

    # Index scalar or tensor std_1/std_2/correlations
    s1_k = std_1.squeeze()[k_index] if std_1.numel() > 1 else std_1.item()
    s2_k = std_2.squeeze()[k_index] if std_2.numel() > 1 else std_2.item()
    rho_k = correlations.squeeze()[k_index] if correlations.numel() > 1 else correlations.item()

    cov[0, 0] = s1_k**2
    cov[1, 1] = s2_k**2
    cov[0, 1], cov[1, 0] = rho_k * s1_k * s2_k, rho_k * s1_k * s2_k


    # Ensure means and stds for torch.normal are tensors
    mean_tensor = torch.tensor([0.0, 0.0], device=y_hat.device)
    std_tensor = torch.tensor([1.0, 1.0], device=y_hat.device)
    x = torch.normal(mean=mean_tensor, std=std_tensor)

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

# purely autoregressive model with LSTM
class HandWritingPredictionNet(nn.Module):

    def __init__(self, hidden_size=400, n_layers=3, output_size=121, input_size=3):
        super(HandWritingPredictionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.LSTM_layers = nn.ModuleList()
        self.LSTM_layers.append(nn.LSTM(input_size, hidden_size, 1, batch_first=True))
        for i in range(n_layers - 1):
            self.LSTM_layers.append(
                nn.LSTM(input_size + hidden_size, hidden_size, 1, batch_first=True)
            )

        self.output_layer = nn.Linear(n_layers * hidden_size, output_size)

        # self.init_weight()

    def forward(self, inputs, initial_hidden):
        hiddens = []
        hidden_cell_state = []  # list of tuple(hn,cn) for each layer
        output, hidden = self.LSTM_layers[0](
            inputs, (initial_hidden[0][0:1], initial_hidden[1][0:1])
        )
        hiddens.append(output)
        hidden_cell_state.append(hidden)
        for i in range(1, self.n_layers):
            inp = torch.cat((inputs, output), dim=2)
            output, hidden = self.LSTM_layers[i](
                inp, (initial_hidden[0][i: i + 1], initial_hidden[1][i: i + 1])
            )
            hiddens.append(output)
            hidden_cell_state.append(hidden)
        inp = torch.cat(hiddens, dim=2)
        y_hat = self.output_layer(inp)

        hidden_states = tuple(torch.cat(layer_states, dim=0) for layer_states in zip(*hidden_cell_state))


        return y_hat, hidden_states

    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        return initial_hidden

    def init_weight(self):
        k = math.sqrt(1.0 / self.hidden_size)
        for layer in self.LSTM_layers:
             for param in layer.parameters():
                  nn.init.uniform_(param, a=-k, b=k)


        nn.init.uniform_(self.output_layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def generate(self, inp, hidden, seq_len, bias, style=None, prime=False):
        gen_seq = []
        batch_size = inp.shape[0] # Use input batch size

        with torch.no_grad():
            current_hidden = hidden
            if prime:
                if style is None:
                     raise ValueError("Style tensor must be provided when prime=True")
                # Process the priming sequence
                y_hat_prime, current_hidden = self.forward(style, current_hidden)

                # Use the last output of the priming sequence to sample the first generated point
                last_output = y_hat_prime[:, -1, :] # Shape: [batch_size, output_size]
                # Sample for each item in the batch
                inp = sample_batch_from_out_dist(last_output, bias) # Shape: [batch_size, 1, 3]

                # Append the priming sequence to gen_seq
                # Ensure style is correctly formatted [batch_size, seq_len, 3]
                gen_seq.append(style)

            # else: # Not priming, inp is likely zeros initially
                 # No change needed here, initial inp and hidden are used

            # Generation loop
            for _ in range(seq_len):
                # Pass current input and hidden state
                y_hat, current_hidden = self.forward(inp, current_hidden)

                # Sample the next point using the output
                # y_hat is [batch_size, 1, output_size], need [batch_size, output_size] for sampling
                next_point_dist = y_hat.squeeze(1)
                inp = sample_batch_from_out_dist(next_point_dist, bias) # inp is now [batch_size, 1, 3]

                # Append the generated point
                gen_seq.append(inp)


        # Concatenate along the sequence dimension (dim=1)
        gen_seq_tensor = torch.cat(gen_seq, dim=1)
        gen_seq_np = gen_seq_tensor.detach().cpu().numpy()

        return gen_seq_np


# --- Positional Encoding (Modified for batch_first=True) ---
class PositionalEncoding(nn.Module):
    # From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # Adapted for batch_first=True
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
         # Check if d_model is odd, handle accordingly
        if d_model % 2 != 0:
            # Handle odd d_model: Calculate up to d_model-1 for cosine
            pe[:, 1::2] = torch.cos(position * div_term[:-1]) # Use div_term up to the second to last element
        else:
             pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension placeholder: [1, max_len, d_model] for broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Shape [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # self.pe buffer shape is [1, max_len, d_model]
        # x shape is [batch_size, seq_len, d_model]
        # Slice pe to match the seq_len of x: self.pe[:, :x.size(1)] -> [1, seq_len, d_model]
        # Add it to x. The batch dimension (1) will broadcast across x's batch_size.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- Handwriting Synthesis Network (Modified for batch_first) ---
class HandWritingSynthesisNet(nn.Module):
    def __init__(self, hidden_size=400, n_layers=3, output_size=121, input_size=3,
                 vocab_size=80, embedding_dim=128, nhead=8, transformer_layers=2,
                 dropout=0.1, window_size=10):
        super(HandWritingSynthesisNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.transformer_layers = transformer_layers
        self.dropout = dropout
        self.window_size = window_size

        # end of sequence
        self.EOS = False

        # Text Encoding Path
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use the modified PositionalEncoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # IMPORTANT: Ensure TransformerEncoderLayer uses batch_first=True
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, hidden_size * 2, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, transformer_layers)

        # Attention Mechanism (Graves' style window)
        self.attention_linear = nn.Linear(hidden_size, 3 * window_size) # Predict alpha, beta, kappa for window

        # LSTM Layers
        self.lstm_1 = nn.LSTM(input_size + embedding_dim, hidden_size, 1, batch_first=True)
        lstm_input_dim = input_size + embedding_dim + hidden_size
        self.lstm_2 = nn.LSTM(lstm_input_dim, hidden_size, 1, batch_first=True)
        self.lstm_3 = nn.LSTM(lstm_input_dim, hidden_size, 1, batch_first=True)

        # Output Layer
        self.output_layer = nn.Linear(n_layers * hidden_size + embedding_dim, output_size)

        self._phi = None


    def _calculate_window_vector(self, h_t, text_encoding, prev_kappa, text_mask=None):
        """
        Calculates the attention window parameters (alpha, beta, kappa)
        and the resulting window vector (context vector w_t).
        h_t: Hidden state of the first LSTM layer [batch, hidden_size]
        text_encoding: Encoded text sequence [batch, text_len, embedding_dim] <--- Shape confirmed
        prev_kappa: Previous kappa values [batch, window_size]
        text_mask: Mask for the text sequence [batch, text_len] (1 for tokens, 0 for padding)
        """
        batch_size, text_len, _ = text_encoding.shape # text_encoding is Batch First

        window_params = self.attention_linear(h_t)
        alpha_hat, beta_hat, kappa_hat = torch.split(window_params, self.window_size, dim=1)

        alpha = torch.exp(alpha_hat)
        beta = torch.exp(beta_hat)
        kappa = prev_kappa + torch.exp(kappa_hat)

        # u shape: [1, text_len, 1]
        u = torch.arange(text_len, device=h_t.device).float().view(1, -1, 1)
        # Unsqueeze alpha, beta, kappa: [batch, 1, window_size]
        alpha_unsqueezed = alpha.unsqueeze(1)
        beta_unsqueezed = beta.unsqueeze(1)
        kappa_unsqueezed = kappa.unsqueeze(1)

        # exponent shape: [batch, text_len, window_size]
        exponent = -beta_unsqueezed * (kappa_unsqueezed - u)**2

        # phi shape: [batch, text_len]
        phi = torch.sum(alpha_unsqueezed * torch.exp(exponent), dim=2)

        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True

        if text_mask is not None:
             phi = phi * text_mask # Apply mask

        phi_sum = torch.sum(phi, dim=1, keepdim=True) + 1e-8
        phi_normalized = phi / phi_sum # Shape: [batch, text_len]

        # bmm input shapes:
        # phi_normalized.unsqueeze(1): [batch, 1, text_len]
        # text_encoding:             [batch, text_len, embedding_dim]
        # Result:                    [batch, 1, embedding_dim] -> squeeze -> [batch, embedding_dim]
        window_vector = torch.bmm(phi_normalized.unsqueeze(1), text_encoding).squeeze(1)

        return window_vector, phi_normalized, kappa


    def forward(self, stroke_inputs, text_tokens, text_mask, initial_hidden, initial_context):
        # ... (other parts remain the same) ...
        batch_size, seq_len, _ = stroke_inputs.shape
        text_len = text_tokens.shape[1]
        device = stroke_inputs.device

        # 1. Encode Text using Transformer (Batch First)
        # text_tokens shape: [batch, text_len]
        # text_mask shape: [batch, text_len]
        embedded_text = self.embedding(text_tokens) * math.sqrt(self.embedding_dim) # [batch, text_len, emb_dim]
        # Pass through positional encoder (expects/returns batch first)
        embedded_text_pos = self.pos_encoder(embedded_text) # [batch, text_len, emb_dim]

        # Create src_key_padding_mask for TransformerEncoder
        # Transformer expects mask where True indicates padding
        src_key_padding_mask = (text_mask == 0) # Shape [batch, text_len]

        # Pass through transformer encoder (expects/returns batch first)
        # Input shape: [batch, text_len, emb_dim]
        # Mask shape: [batch, text_len]
        # Output shape: [batch, text_len, emb_dim]
        text_encoding = self.transformer_encoder(embedded_text_pos, src_key_padding_mask=src_key_padding_mask)

        # --- Rest of the forward pass remains the same ---
        # Initialize attention state
        prev_kappa = initial_context[0]

        # Initialize lists to store outputs and states
        all_y_hat = []
        all_phi = []
        all_hidden_h = [[] for _ in range(self.n_layers)]
        all_hidden_c = [[] for _ in range(self.n_layers)]

        h_prev = [s.squeeze(0) for s in initial_hidden[0].split(1, dim=0)]
        c_prev = [s.squeeze(0) for s in initial_hidden[1].split(1, dim=0)]

        # 2. Process Stroke Sequence step-by-step
        for t in range(seq_len):
            stroke_input_t = stroke_inputs[:, t, :]

            # Calculate window vector using text_encoding [batch, text_len, emb_dim]
            window_vector_t, phi_t, current_kappa = self._calculate_window_vector(
                h_prev[0], text_encoding, prev_kappa, text_mask
            )
            prev_kappa = current_kappa
            all_phi.append(phi_t) # Store attention

            # --- LSTM Layers ---
            # Layer 1
            lstm1_input = torch.cat((stroke_input_t, window_vector_t), dim=1)
            h1, c1 = h_prev[0].unsqueeze(0), c_prev[0].unsqueeze(0)
            output1, (h1_new, c1_new) = self.lstm_1(lstm1_input.unsqueeze(1), (h1, c1))
            output1 = output1.squeeze(1)
            h_prev[0], c_prev[0] = h1_new.squeeze(0), c1_new.squeeze(0)
            all_hidden_h[0].append(h_prev[0])
            all_hidden_c[0].append(c_prev[0])

            # Layer 2
            lstm2_input = torch.cat((stroke_input_t, window_vector_t, output1), dim=1)
            h2, c2 = h_prev[1].unsqueeze(0), c_prev[1].unsqueeze(0)
            output2, (h2_new, c2_new) = self.lstm_2(lstm2_input.unsqueeze(1), (h2, c2))
            output2 = output2.squeeze(1)
            h_prev[1], c_prev[1] = h2_new.squeeze(0), c2_new.squeeze(0)
            all_hidden_h[1].append(h_prev[1])
            all_hidden_c[1].append(c_prev[1])

            # Layer 3
            lstm3_input = torch.cat((stroke_input_t, window_vector_t, output2), dim=1)
            h3, c3 = h_prev[2].unsqueeze(0), c_prev[2].unsqueeze(0)
            output3, (h3_new, c3_new) = self.lstm_3(lstm3_input.unsqueeze(1), (h3, c3))
            output3 = output3.squeeze(1)
            h_prev[2], c_prev[2] = h3_new.squeeze(0), c3_new.squeeze(0)
            all_hidden_h[2].append(h_prev[2])
            all_hidden_c[2].append(c_prev[2])

            # --- Output Layer ---
            final_lstm_input = torch.cat((output1, output2, output3, window_vector_t), dim=1)
            y_hat_t = self.output_layer(final_lstm_input)
            all_y_hat.append(y_hat_t)


        # Stack results across the sequence dimension
        y_hat = torch.stack(all_y_hat, dim=1) # [batch, seq_len, output_size]
        self._phi = torch.stack(all_phi, dim=1) # [batch, seq_len, text_len]

        last_h = torch.stack(h_prev, dim=0) # [n_layers, batch, hidden_size]
        last_c = torch.stack(c_prev, dim=0) # [n_layers, batch, hidden_size]
        final_hidden_state = (last_h, last_c)

        final_context = (prev_kappa, window_vector_t) # Last context

        return y_hat, final_hidden_state, final_context


    # --- init_hidden remains the same ---
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        initial_hidden = (h0, c0)

        initial_kappa = torch.zeros(batch_size, self.window_size, device=device)
        initial_window_vector = torch.zeros(batch_size, self.embedding_dim, device=device)
        initial_context = (initial_kappa, initial_window_vector)

        return initial_hidden, initial_context


    @torch.no_grad()
    def generate(self, inp, text, text_mask, hidden, context, bias, prime=False, prime_text=None, prime_mask=None, max_len=1000):
        batch_size = text.shape[0]
        device = text.device
        gen_seq = []
        phi_during_gen = [] # Store phi separately during generation

        # 1. Encode Target Text (Batch First)
        embedded_text = self.embedding(text) * math.sqrt(self.embedding_dim)
        embedded_text_pos = self.pos_encoder(embedded_text) # Batch first
        src_key_padding_mask = (text_mask == 0)
        # Transformer expects/returns batch first
        text_encoding = self.transformer_encoder(embedded_text_pos, src_key_padding_mask=src_key_padding_mask) # [batch, text_len, emb_dim]

        # --- Priming Phase (if enabled) ---
        current_hidden = hidden
        current_kappa, current_window_vec = context

        if prime:
            if inp is None or prime_text is None or prime_mask is None:
                 raise ValueError("inp (prime_seq), prime_text, and prime_mask must be provided when prime=True")

            prime_seq_len = inp.shape[1]
            print(f"Priming with sequence of length {prime_seq_len}...")

            self.EOS = False

            # Encode priming text (batch first) - not strictly needed unless attention uses it
            # embedded_prime_text = self.embedding(prime_text) * math.sqrt(self.embedding_dim)
            # embedded_prime_text_pos = self.pos_encoder(embedded_prime_text) # Batch first
            # prime_src_key_padding_mask = (prime_mask == 0)
            # prime_text_encoding = self.transformer_encoder(embedded_prime_text_pos, src_key_padding_mask=prime_src_key_padding_mask)

            gen_seq.append(inp) # Append prime sequence

            h_prev = [s.squeeze(0) for s in current_hidden[0].split(1, dim=0)]
            c_prev = [s.squeeze(0) for s in current_hidden[1].split(1, dim=0)]

            # Process priming strokes
            for t in range(prime_seq_len):
                stroke_input_t = inp[:, t, :]
                # Use TARGET text encoding for attention during priming
                window_vector_t, phi_t, current_kappa = self._calculate_window_vector(
                    h_prev[0], text_encoding, current_kappa, text_mask
                )
                # Maybe store priming phi? phi_during_prime.append(phi_t)

                # --- LSTM Layers (same as forward pass loop) ---
                # Layer 1
                lstm1_input = torch.cat((stroke_input_t, window_vector_t), dim=1)
                h1, c1 = h_prev[0].unsqueeze(0), c_prev[0].unsqueeze(0)
                output1, (h1_new, c1_new) = self.lstm_1(lstm1_input.unsqueeze(1), (h1, c1))
                h_prev[0], c_prev[0] = h1_new.squeeze(0), c1_new.squeeze(0)
                output1 = output1.squeeze(1)
                # Layer 2
                lstm2_input = torch.cat((stroke_input_t, window_vector_t, output1), dim=1)
                h2, c2 = h_prev[1].unsqueeze(0), c_prev[1].unsqueeze(0)
                output2, (h2_new, c2_new) = self.lstm_2(lstm2_input.unsqueeze(1), (h2, c2))
                h_prev[1], c_prev[1] = h2_new.squeeze(0), c2_new.squeeze(0)
                output2 = output2.squeeze(1)
                # Layer 3
                lstm3_input = torch.cat((stroke_input_t, window_vector_t, output2), dim=1)
                h3, c3 = h_prev[2].unsqueeze(0), c_prev[2].unsqueeze(0)
                output3, (h3_new, c3_new) = self.lstm_3(lstm3_input.unsqueeze(1), (h3, c3))
                h_prev[2], c_prev[2] = h3_new.squeeze(0), c3_new.squeeze(0)
                output3 = output3.squeeze(1)

                if t == prime_seq_len - 1:
                     final_lstm_input = torch.cat((output1, output2, output3, window_vector_t), dim=1)
                     last_y_hat_prime = self.output_layer(final_lstm_input)

            # Update hidden state and sample first generated point
            current_hidden = (torch.stack(h_prev, dim=0), torch.stack(c_prev, dim=0))
            inp = sample_batch_from_out_dist(last_y_hat_prime, bias)

        else:
            # Not priming
            if inp.shape[1] != 1:
                 raise ValueError("Initial input 'inp' must have sequence length 1 when not priming.")
            print("Starting generation without priming...")


        # --- Generation Loop ---
        print(f"Generating sequence of max_len {max_len}...")
        h_prev = [s.squeeze(0) for s in current_hidden[0].split(1, dim=0)]
        c_prev = [s.squeeze(0) for s in current_hidden[1].split(1, dim=0)]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        steps_taken = 0

        while steps_taken < max_len and not self.EOS:
        # for _ in range(max_len):
            stroke_input_t = inp.squeeze(1) # [batch, 3]

            # Calculate window vector using TARGET text encoding
            window_vector_t, phi_t, current_kappa = self._calculate_window_vector(
                h_prev[0], text_encoding, current_kappa, text_mask
            )
            phi_during_gen.append(phi_t) # Store attention weights

            # --- LSTM Layers (same as priming loop) ---
             # Layer 1
            lstm1_input = torch.cat((stroke_input_t, window_vector_t), dim=1)
            h1, c1 = h_prev[0].unsqueeze(0), c_prev[0].unsqueeze(0)
            output1, (h1_new, c1_new) = self.lstm_1(lstm1_input.unsqueeze(1), (h1, c1))
            h_prev[0], c_prev[0] = h1_new.squeeze(0), c1_new.squeeze(0)
            output1 = output1.squeeze(1)
            # Layer 2
            lstm2_input = torch.cat((stroke_input_t, window_vector_t, output1), dim=1)
            h2, c2 = h_prev[1].unsqueeze(0), c_prev[1].unsqueeze(0)
            output2, (h2_new, c2_new) = self.lstm_2(lstm2_input.unsqueeze(1), (h2, c2))
            h_prev[1], c_prev[1] = h2_new.squeeze(0), c2_new.squeeze(0)
            output2 = output2.squeeze(1)
            # Layer 3
            lstm3_input = torch.cat((stroke_input_t, window_vector_t, output2), dim=1)
            h3, c3 = h_prev[2].unsqueeze(0), c_prev[2].unsqueeze(0)
            output3, (h3_new, c3_new) = self.lstm_3(lstm3_input.unsqueeze(1), (h3, c3))
            h_prev[2], c_prev[2] = h3_new.squeeze(0), c3_new.squeeze(0)
            output3 = output3.squeeze(1)

            # --- Output Layer ---
            final_lstm_input = torch.cat((output1, output2, output3, window_vector_t), dim=1)
            y_hat_t = self.output_layer(final_lstm_input)

            # Sample next point
            next_inp = sample_batch_from_out_dist(y_hat_t, bias) # [batch, 1, 3]

            # Append the point that was *input* to this step
            gen_seq.append(inp) # inp shape [batch, 1, 3]

            inp = next_inp # Update input for next step
            steps_taken += 1

            if steps_taken % 100 == 0:
                 print(f"Generated {steps_taken} steps...")


        if not gen_seq:
            return np.zeros((batch_size, 0, 3))

        gen_seq_tensor = torch.cat(gen_seq, dim=1)
        # Store the collected attention weights
        self._phi = torch.stack(phi_during_gen, dim=1) if phi_during_gen else None # [batch, generated_len, text_len]

        print(f"Finished generation after {steps_taken} steps.")
        gen_seq_np = gen_seq_tensor.detach().cpu().numpy()
        return gen_seq_np