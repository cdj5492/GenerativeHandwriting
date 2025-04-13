from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, Categorical

# Define the cell state as a named tuple.
LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)

class LSTMAttentionCell(nn.Module):
    def __init__(self,
                 input_size,
                 lstm_size,
                 num_attn_mixture_components,
                 attention_values,           # tensor of shape [batch, char_len, alphabet_size]
                 attention_values_lengths,   # tensor of shape [batch] (each element <= char_len)
                 num_output_mixture_components,
                 bias,
                 device='cuda',
                 ):
        """
        Args:
            input_size: Dimensionality of external input.
            lstm_size: Hidden size for each LSTM cell.
            num_attn_mixture_components: Number of mixtures for the attention.
            attention_values: Tensor of shape [batch, char_len, alphabet_size] used for attention.
            attention_values_lengths: Tensor of shape [batch] giving valid lengths in attention_values.
            num_output_mixture_components: Number of mixtures for the output GMM.
            bias: A tensor bias applied in the output parameter calculation (e.g. for adjusting mixture weights).
        """
        super(LSTMAttentionCell, self).__init__()
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values  # fixed tensor provided at initialization
        self.attention_values_lengths = attention_values_lengths  # shape [batch]
        # In TF, window_size is the static size of dimension 2 of attention_values.
        # Here, we assume attention_values has shape [batch, char_len, alphabet_size]
        self.alphabet_size = attention_values.size(2)
        self.char_len = attention_values.size(1)
        self.batch_size = attention_values.size(0)
        self.num_output_mixture_components = num_output_mixture_components
        # Output units for the GMM part (6 parameters per mixture plus one Bernoulli parameter)
        self.output_units = 6 * self.num_output_mixture_components + 1
        self.bias = bias

        # We need to know the input size of the external inputs.
        self.input_size = input_size

        # LSTM 1: receives concatenation of state.w (dim: alphabet_size) and inputs (dim: input_size)
        self.lstm1 = nn.LSTMCell(input_size=self.alphabet_size + self.input_size, hidden_size=self.lstm_size).to(device)
        # Attention layer: from concat(state.w, inputs, s1_out) to 3 * num_attn_mixture_components.
        self.attention_layer = nn.Linear(self.alphabet_size + self.input_size + self.lstm_size,
                                         3 * self.num_attn_mixture_components).to(device)
        # LSTM 2: receives [inputs, s1_out, w] where w has dim alphabet_size.
        self.lstm2 = nn.LSTMCell(input_size=self.input_size + self.lstm_size + self.alphabet_size,
                                  hidden_size=self.lstm_size).to(device)
        # LSTM 3: receives [inputs, s2_out, w]
        self.lstm3 = nn.LSTMCell(input_size=self.input_size + self.lstm_size + self.alphabet_size,
                                  hidden_size=self.lstm_size).to(device)
        # Output layer (for the GMM parameters) applied to state.h3.
        self.gmm_layer = nn.Linear(self.lstm_size, self.output_units).to(device)

    def state_size(self):
        # Returns a tuple of state sizes (for reference only)
        return LSTMAttentionCellState(
            h1=self.lstm_size,
            c1=self.lstm_size,
            h2=self.lstm_size,
            c2=self.lstm_size,
            h3=self.lstm_size,
            c3=self.lstm_size,
            alpha=self.num_attn_mixture_components,
            beta=self.num_attn_mixture_components,
            kappa=self.num_attn_mixture_components,
            w=self.alphabet_size,
            phi=self.char_len,
        )

    def zero_state(self, batch_size, device):
        zeros = lambda dim: torch.zeros(batch_size, dim, device=device)
        return LSTMAttentionCellState(
            h1=zeros(self.lstm_size),
            c1=zeros(self.lstm_size),
            h2=zeros(self.lstm_size),
            c2=zeros(self.lstm_size),
            h3=zeros(self.lstm_size),
            c3=zeros(self.lstm_size),
            alpha=zeros(self.num_attn_mixture_components),
            beta=zeros(self.num_attn_mixture_components),
            kappa=zeros(self.num_attn_mixture_components),
            w=zeros(self.alphabet_size),
            phi=zeros(self.char_len)
        )

    def forward(self, inputs, state):
        """
        Args:
            inputs: external input tensor of shape [batch, input_size].
            state: a LSTMAttentionCellState namedtuple containing previous states.
        Returns:
            output: tensor of shape [batch, lstm_size] (from LSTM 3).
            new_state: a LSTMAttentionCellState with updated states.
        """
        # LSTM 1: concatenate previous attention output w and inputs.
        s1_in = torch.cat([state.w, inputs], dim=1)
        h1_new, c1_new = self.lstm1(s1_in, (state.h1, state.c1))
        
        # Attention mechanism.
        attention_inputs = torch.cat([state.w, inputs, h1_new], dim=1)
        attention_params = self.attention_layer(attention_inputs)
        # Apply softplus and split into alpha, beta, kappa_update.
        attn_out = F.softplus(attention_params)
        alpha, beta, kappa_update = torch.split(attn_out, self.num_attn_mixture_components, dim=1)
        # Update kappa: add a scaled update.
        kappa_new = state.kappa + kappa_update / 25.0
        beta = torch.clamp(beta, min=0.01)
        # Expand dims for broadcasting: shape becomes [batch, num_attn_mixture_components, 1]
        alpha_exp = alpha.unsqueeze(2)
        beta_exp = beta.unsqueeze(2)
        kappa_exp = kappa_new.unsqueeze(2)

        # Create u: tensor with shape [1, 1, char_len]
        u = torch.arange(self.char_len, device=inputs.device).float().view(1, 1, self.char_len)
        # Expand to [batch, num_attn_mixture_components, char_len]
        u = u.expand(inputs.size(0), self.num_attn_mixture_components, self.char_len)
        # Compute phi_flat: sum over mixtures. Resulting shape [batch, char_len]
        phi_flat = torch.sum(alpha_exp * torch.exp(- (kappa_exp - u) ** 2 / beta_exp), dim=1)
        # In TF, phi is expanded then used with attention_values.
        phi = phi_flat.unsqueeze(2)  # shape: [batch, char_len, 1]
        # Create sequence mask for attention_values_lengths.

        # Compute phi_flat: sum over mixtures. Resulting shape [batch, char_len]
        phi_flat = torch.sum(alpha_exp * torch.exp(- (kappa_exp - u) ** 2 / beta_exp), dim=1)
        # In TF, phi is expanded then used with attention_values.
        phi = phi_flat.unsqueeze(2)  # shape: [batch, char_len, 1]

        # --- FIX: Replicate attention values & lengths if batch sizes differ ---
        # If the batch size in attention_values is different from the current inputs batch size,
        # replicate the first attention sample for all inputs.
        if self.attention_values.size(0) != inputs.size(0):
            attn_values = self.attention_values[:1].expand(inputs.size(0), self.char_len, self.alphabet_size)
            attn_lengths = self.attention_values_lengths[:1].expand(inputs.size(0))
        else:
            attn_values = self.attention_values
            attn_lengths = self.attention_values_lengths

        # Create sequence mask for attn_lengths.
        mask = (torch.arange(self.char_len, device=inputs.device)
                .unsqueeze(0).expand(inputs.size(0), self.char_len)
                < attn_lengths.unsqueeze(1)).float()
        mask = mask.unsqueeze(2)  # shape: [batch, char_len, 1]

        # Compute weighted attention vector w using the replicated attention values.
        w = torch.sum(phi * attn_values * mask, dim=1)  # shape: [batch, alphabet_size]

        mask = mask.unsqueeze(2)  # shape: [batch, char_len, 1]
        # Compute weighted attention vector w.
        # attention_values has shape [batch, char_len, alphabet_size]
        w = torch.sum(phi * self.attention_values * mask, dim=1)  # shape: [batch, alphabet_size]

        # LSTM 2: input is [inputs, h1_new, w]
        s2_in = torch.cat([inputs, h1_new, w], dim=1)
        h2_new, c2_new = self.lstm2(s2_in, (state.h2, state.c2))
        
        # LSTM 3: input is [inputs, h2_new, w]
        s3_in = torch.cat([inputs, h2_new, w], dim=1)
        h3_new, c3_new = self.lstm3(s3_in, (state.h3, state.c3))

        new_state = LSTMAttentionCellState(
            h1=h1_new,
            c1=c1_new,
            h2=h2_new,
            c2=c2_new,
            h3=h3_new,
            c3=c3_new,
            alpha=alpha,     # save the unexpanded values (alpha_flat)
            beta=beta,       # beta_flat
            kappa=kappa_new, # updated kappa
            w=w,
            phi=phi_flat     # phi_flat is [batch, char_len]
        )
        return h3_new, new_state

    def output_function(self, state):
        """
        Compute the output distribution parameters and sample an output.
        Returns:
            A tensor of shape [batch, 3] where the first two elements are the selected coordinate
            and the third element is the Bernoulli sample.
        """
        gmm_params = self.gmm_layer(state.h3)
        pis, mus, sigmas, rhos, es = self._parse_parameters(gmm_params)
        # Split mus into two parts and stack them to get shape [batch, num_mixtures, 2]
        mu1, mu2 = torch.split(mus, self.num_output_mixture_components, dim=1)
        mus_stacked = torch.stack([mu1, mu2], dim=2)
        # Similarly, split sigmas into sigma1 and sigma2.
        sigma1, sigma2 = torch.split(sigmas, self.num_output_mixture_components, dim=1)
        # Build covariance matrices for each mixture component.
        # They will have shape [batch, num_mixtures, 2, 2].
        cov11 = sigma1 ** 2
        cov22 = sigma2 ** 2
        cov12 = rhos * sigma1 * sigma2
        # Construct full covariance matrix.
        covar_matrix = torch.stack([cov11, cov12, cov12, cov22], dim=2)
        covar_matrix = covar_matrix.view(-1, self.num_output_mixture_components, 2, 2)
        
        # Create distributions.
        # For the multivariate normals, we construct one for each mixture.
        mvn = MultivariateNormal(loc=mus_stacked, covariance_matrix=covar_matrix)
        b = Bernoulli(probs=es)
        c = Categorical(probs=pis)
        # Sample from each distribution.
        sampled_e = b.sample()  # shape: [batch, 1] (if es is [batch, 1]) or [batch] if scalar per sample.
        sampled_coords = mvn.sample()  # shape: [batch, num_mixtures, 2]
        sampled_idx = c.sample()       # shape: [batch]
        # Gather the coordinates corresponding to the sampled mixture component.
        idx = sampled_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
        coords = torch.gather(sampled_coords, dim=1, index=idx).squeeze(1)
        return torch.cat([coords, sampled_e.float().unsqueeze(1)], dim=1)

    def termination_condition(self, state):
        """
        Checks if termination conditions are met.
        Returns:
            A boolean tensor of shape [batch] indicating which samples should terminate.
        """
        # Compute the index of the most attended character.
        # phi has shape [batch, char_len]; argmax along char_len.
        char_idx = torch.argmax(state.phi, dim=1).int()
        # final_char if the attended index is at least (length - 1)
        final_char = char_idx >= (self.attention_values_lengths - 1)
        # past_final_char if the index is greater than or equal to the length.
        past_final_char = char_idx >= self.attention_values_lengths
        output = self.output_function(state)
        # Assume the Bernoulli sample is at index 2.
        es = output[:, 2].int()
        is_eos = es == 1
        # Terminate if either (final_char and is_eos) or past_final_char.
        return (final_char & is_eos) | past_final_char

    def _parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4):
        """
        Parse the output parameters for the GMM.
        Splits the input tensor into mixture coefficients, means, sigmas, correlation coefficients, and Bernoulli parameters.
        """
        splits = torch.split(gmm_params, 
                             [self.num_output_mixture_components,
                              2 * self.num_output_mixture_components,
                              self.num_output_mixture_components,
                              2 * self.num_output_mixture_components,
                              1], dim=1)
        pis, sigmas, rhos, mus, es = splits
        # Adjust with bias.
        bias_expanded = self.bias.unsqueeze(1)  # assume self.bias is of shape [batch] or a scalar tensor
        pis = pis * (1 + bias_expanded)
        sigmas = sigmas - bias_expanded

        pis = F.softmax(pis, dim=1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        es = torch.where(es < 0.01, torch.zeros_like(es), es)
        return pis, mus, sigmas, rhos, es

    def dynamic_rnn(self, inputs, sequence_lengths, initial_state, device):
        batch_size, max_time, _ = inputs.size()
        state = initial_state
        outputs = []
        for t in range(max_time):
            input_t = inputs[:, t, :]
            output, state = self(input_t, state)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, state
