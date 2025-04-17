
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli
from utils.model_utils import stable_softmax

def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y_parts = torch.split(y_hat, split_sizes, dim=-1)

    eos_prob = torch.sigmoid(y_parts[0])
    mixture_weights = stable_softmax(y_parts[1] * (1 + bias), dim=-1)
    mu_1 = y_parts[2]
    mu_2 = y_parts[3]
    std_1 = torch.exp(y_parts[4] - bias)
    std_2 = torch.exp(y_parts[5] - bias)
    correlations = torch.tanh(y_parts[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1).squeeze()
    mu_k = torch.cat((mu_1[:, K], mu_2[:, K]), dim=-1)

    std1 = std_1[:, K]
    std2 = std_2[:, K]
    corr = correlations[:, K]

    cov = torch.zeros(2, 2, device=y_hat.device)
    cov[0, 0] = std1.pow(2)
    cov[1, 1] = std2.pow(2)
    cov[0, 1] = cov[1, 0] = corr * std1 * std2

    x = torch.randn(2, device=y_hat.device)
    Z = mu_k.squeeze() + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 3)
    sample[0, 0] = eos_sample.item()
    sample[0, 1:] = Z
    return sample

def sample_batch_from_out_dist(y_hat, bias):
    batch_size = y_hat.shape[0]
    split_sizes = [1] + [20] * 6
    y_parts = torch.split(y_hat, split_sizes, dim=-1)

    eos_prob = torch.sigmoid(y_parts[0])
    mixture_weights = stable_softmax(y_parts[1] * (1 + bias), dim=-1)
    mu_1 = y_parts[2]
    mu_2 = y_parts[3]
    std_1 = torch.exp(y_parts[4] - bias)
    std_2 = torch.exp(y_parts[5] - bias)
    correlations = torch.tanh(y_parts[6])

    bernoulli_dist = bernoulli.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights.squeeze(), 1).squeeze()
    idx = torch.arange(batch_size, device=y_hat.device)

    mu_k = torch.zeros(batch_size, 2, device=y_hat.device)
    mu_k[:, 0] = mu_1[idx, K]
    mu_k[:, 1] = mu_2[idx, K]

    cov = torch.zeros(batch_size, 2, 2, device=y_hat.device)
    cov[:, 0, 0] = std_1[idx, K].pow(2)
    cov[:, 1, 1] = std_2[idx, K].pow(2)
    temp = correlations[idx, K] * std_1[idx, K] * std_2[idx, K]
    cov[:, 0, 1] = temp
    cov[:, 1, 0] = temp

    X = torch.randn(batch_size, 2, 1, device=y_hat.device)
    Z = mu_k + torch.matmul(cov, X).squeeze(-1)

    sample = y_hat.new_zeros(batch_size, 3)
    sample[:, 0:1] = eos_sample
    sample[:, 1:] = Z
    return sample

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HandWritingPredictionNet(nn.Module):
    def __init__(self, hidden_size=128, n_layers=3, output_size=121, input_size=3, max_seq_len=1000):
        super(HandWritingPredictionNet, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, initial_hidden=None):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        x = self.transformer_encoder(x, mask=mask)
        x = x.transpose(0, 1)
        y_hat = self.output_layer(x)
        return y_hat, None

    def init_hidden(self, batch_size, device):
        return None

    def generate(self, inp, hidden, seq_len, bias, style=None, prime=False):
        if prime and style is not None:
            current_seq = style
        else:
            current_seq = inp
        gen_seq = [current_seq]
        batch_size = inp.size(0)
        for i in range(seq_len):
            y_hat, _ = self.forward(current_seq)
            last_output = y_hat[:, -1, :]
            next_point = sample_batch_from_out_dist(last_output, bias).unsqueeze(1)
            current_seq = torch.cat((current_seq, next_point), dim=1)
            gen_seq.append(next_point)
            if (next_point[:, 0, 0] > 0.5).all():
                break
        gen_seq = torch.cat(gen_seq, dim=1)
        return gen_seq.detach().cpu().numpy()

class HandWritingSynthesisNet(nn.Module):
    def __init__(self, hidden_size=128, n_layers=3, output_size=121, window_size=77, max_seq_len=1000):
        super(HandWritingSynthesisNet, self).__init__()
        self.input_proj = nn.Linear(3, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        self.text_embedding = nn.Embedding(window_size, hidden_size)
        self.text_pos_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        return None, None, None

    def forward(self, inputs, text, text_mask, initial_hidden=None, prev_window_vec=None, prev_kappa=None, is_map=False):
        tgt = self.input_proj(inputs)
        tgt = self.pos_encoding(tgt).transpose(0, 1)
        mem = self.text_embedding(text)
        mem = self.text_pos_encoding(mem).transpose(0, 1)
        seq_len = tgt.size(0)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device) * float('-inf'), diagonal=1)
        dec_output = self.transformer_decoder(tgt, mem, tgt_mask=tgt_mask)
        dec_output = dec_output.transpose(0, 1)
        y_hat = self.output_layer(dec_output)
        return y_hat, None, None, None

    def generate(self, inp, text, text_mask, prime_text, prime_mask, hidden, window_vector, kappa, bias, is_map=False, prime=False):
        batch_size = inp.size(0)
        mem = self.text_embedding(text)
        mem = self.text_pos_encoding(mem).transpose(0, 1)
        current_seq = inp
        gen_seq = [current_seq]
        for i in range(1000):
            tgt = self.input_proj(current_seq)
            tgt = self.pos_encoding(tgt).transpose(0, 1)
            seq_len = tgt.size(0)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device) * float('-inf'), diagonal=1)
            dec_output = self.transformer_decoder(tgt, mem, tgt_mask=tgt_mask)
            dec_output = dec_output.transpose(0, 1)
            last_output = dec_output[:, -1, :]
            y_hat = self.output_layer(last_output)
            next_point = sample_batch_from_out_dist(y_hat, bias).unsqueeze(1)
            current_seq = torch.cat((current_seq, next_point), dim=1)
            gen_seq.append(next_point)
            if (next_point[:, 0, 0] > 0.5).all():
                break
        gen_seq = torch.cat(gen_seq, dim=1)
        return gen_seq.detach().cpu().numpy()
