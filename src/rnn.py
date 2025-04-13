import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import drawing
from dataset import NumpyDataset
from lstm_attention_cell import LSTMAttentionCell  # PyTorch attention cell

# better traceback
import rich.traceback
rich.traceback.install()

# helper function to run the RNN
def rnn_free_run(cell, sequence_length, initial_state, initial_input, device):
    """
    Run the RNN for a specified number of timesteps.
    """
    outputs = []
    state = initial_state
    input = initial_input

    for t in range(sequence_length):
        output, state = cell(input, state)
        outputs.append(output)
        input = output  # Use the output as the next input

    outputs = torch.stack(outputs, dim=1)
    return outputs, state

#########################################
# DataReader using DataLoader
#########################################

def custom_collate_fn(samples):
    """
    Collate a list of samples (dicts) into a batch and perform the time slicing
    operations as in the original batch_generator.
    """
    batch = {}
    # Stack each field into a numpy array then convert to tensor.
    for key in samples[0].keys():
        arr = np.stack([s[key] for s in samples], axis=0)
        batch[key] = torch.tensor(arr)
    
    # Adjust x_len (subtract one) and slice x, y, c accordingly.
    batch['x_len'] = batch['x_len'] - 1
    max_x_len = batch['x_len'].max().item()
    max_c_len = batch['c_len'].max().item()
    
    # y is x shifted by one timestep.
    batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
    batch['x'] = batch['x'][:, :max_x_len, :]
    batch['c'] = batch['c'][:, :max_c_len]
    return batch

class DataReader(object):
    def __init__(self, data_dir, batch_size, num_workers=0):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        data = [np.load(os.path.join(data_dir, f'{col}.npy')) for col in data_cols]
        
        # Create the full dataset
        full_dataset = NumpyDataset(columns=data_cols, data=data)
        # Split train and test (here test is analogous to your validation split)
        self.train_dataset, self.val_dataset = full_dataset.train_test_split(
            train_size=0.95, random_state=2018
        )
        # Use full dataset as test dataset (or adjust as needed)
        self.test_dataset = full_dataset
        
        # Create DataLoaders for each split.
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=custom_collate_fn, drop_last=True
        )
        
        print('Train size:', len(self.train_dataset))
        print('Validation size:', len(self.val_dataset))
        print('Test size:', len(self.test_dataset))

#########################################
# RNN Model
#########################################

class RNNModel(nn.Module):
    def __init__(self, lstm_size, output_mixture_components, attention_mixture_components, **kwargs):
        super(RNNModel, self).__init__()
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        # output_units: 6 parameters per mixture plus 1 extra for the Bernoulli parameter
        self.output_units = self.output_mixture_components * 6 + 1
        self.attention_mixture_components = attention_mixture_components

        # A dense layer to apply at each time step.
        self.gmm_dense = nn.Linear(self.lstm_size, self.output_units)
        # The attention cell will be created dynamically in forward based on the attention inputs.
        self.cell = None

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        # Split z into its parameters.
        splits = torch.split(
            z, [self.output_mixture_components,
                2 * self.output_mixture_components,
                self.output_mixture_components,
                2 * self.output_mixture_components,
                1],
            dim=-1
        )
        pis, sigmas, rhos, mus, es = splits
        pis = F.softmax(pis, dim=-1)
        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def NLL(self, y, lengths, pis, mus, sigmas, rhos, es, eps=1e-8):
        # Split sigmas, y, and mus into their respective components.
        sigma_1, sigma_2 = torch.chunk(sigmas, 2, dim=2)
        y_1, y_2, y_3 = torch.chunk(y, 3, dim=2)
        mu_1, mu_2 = torch.chunk(mus, 2, dim=2)

        norm = 1.0 / (2 * np.pi * sigma_1 * sigma_2 * torch.sqrt(1 - rhos ** 2))
        Z = ((y_1 - mu_1) / sigma_1) ** 2 + ((y_2 - mu_2) / sigma_2) ** 2 - \
            2 * rhos * (y_1 - mu_1) * (y_2 - mu_2) / (sigma_1 * sigma_2)
        gaussian_likelihoods = torch.exp(-Z / (2 * (1 - rhos ** 2))) * norm

        gmm_likelihood = torch.sum(pis * gaussian_likelihoods, dim=2)
        gmm_likelihood = torch.clamp(gmm_likelihood, min=eps)

        bernoulli_likelihood = torch.where(
            y_3.squeeze(-1) == 1, es.squeeze(-1), 1 - es.squeeze(-1)
        )
        nll = -(torch.log(gmm_likelihood) + torch.log(bernoulli_likelihood))

        # Create a mask based on sequence lengths
        max_len = y.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask & (~torch.isnan(nll))
        nll = torch.where(mask, nll, torch.zeros_like(nll))
        num_valid = mask.float().sum(dim=1)

        sequence_loss = torch.sum(nll, dim=1) / torch.clamp(num_valid, min=1.0)
        element_loss = torch.sum(nll) / torch.clamp(torch.sum(num_valid), min=1.0)
        return sequence_loss, element_loss

    def sample(self, cell, num_samples, sample_tsteps, device):
        print("working on regular sample")
        state = cell.zero_state(num_samples, device=device)
        initial_input = torch.cat([
            torch.zeros(num_samples, 2, device=device),
            torch.ones(num_samples, 1, device=device)
        ], dim=1)
        _, sample_seq = rnn_free_run(
            cell=cell,
            sequence_length=sample_tsteps,
            initial_state=state,
            initial_input=initial_input,
            device=device
        )
        return sample_seq

    def primed_sample(self, cell, x_prime, x_prime_len, num_samples, sample_tsteps, device):
        print("working on primed sample")
        state = cell.zero_state(num_samples, device=device)
        primed_outputs, primed_state = cell.dynamic_rnn(
            inputs=x_prime,
            sequence_lengths=x_prime_len,
            initial_state=state,
            device=device
        )
        _, sample_seq = rnn_free_run(
            cell=cell,
            sequence_length=sample_tsteps,
            initial_state=primed_state,
            device=device
        )
        return sample_seq

    def forward(self, x, y, x_len, c, c_len, sample_tsteps, num_samples, prime=False,
                x_prime=None, x_prime_len=None, bias=None, device='cuda'):
        # Prepare attention values as one-hot encoded tensors.
        attention_values = F.one_hot(c, num_classes=len(drawing.alphabet)).float()

        # Create the attention cell; the bias is set or defaulted.
        self.cell = LSTMAttentionCell(
            input_size=x.size(2),
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=attention_values,
            attention_values_lengths=c_len,
            num_output_mixture_components=self.output_mixture_components,
            bias=bias if bias is not None else torch.zeros(num_samples, device=device),
            device=device
        )
        
        batch_size = x.size(0)
        initial_state = self.cell.zero_state(batch_size, device=device)
        
        # Run the RNN over the input sequence.
        outputs, final_state = self.cell.dynamic_rnn(
            inputs=x,
            sequence_lengths=x_len,
            initial_state=initial_state,
            device=device
        )
        
        # Apply a dense layer at each time step.
        params = self.gmm_dense(outputs)
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)
        sequence_loss, loss = self.NLL(y, x_len, pis, mus, sigmas, rhos, es)
        
        if prime:
            assert x_prime is not None and x_prime_len is not None, "x_prime and x_prime_len are required for primed sampling"
            sampled_sequence = self.primed_sample(self.cell, x_prime, x_prime_len, num_samples, sample_tsteps, device)
        else:
            sampled_sequence = self.sample(self.cell, num_samples, sample_tsteps, device)
        
        self.final_state = final_state
        self.sampled_sequence = sampled_sequence
        return loss

#########################################
# Training Script
#########################################

if __name__ == '__main__':
    # Setup: data directory, batch size, etc.
    data_dir = 'data/processed/'
    batch_size = 32
    num_workers = 8  # adjust if you have multiple CPUs

    # Create a DataReader instance that sets up DataLoaders.
    dr = DataReader(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

    # Hyperparameters for the model.
    config = {
        'lstm_size': 400,
        'output_mixture_components': 20,
        'attention_mixture_components': 10,
        # additional hyperparameters as needed...
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(**config).to(device)
    
    # Use RMSprop optimizer as in the original code.
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    num_training_steps = 100000
    sample_tsteps = 100  # example value for sampling timesteps
    num_samples = 10     # number of sequences to sample
    
    model.train()
    step = 0
    # Loop over the DataLoader for training.
    while step < num_training_steps:
        for batch in dr.train_loader:
            # Move batch tensors to the appropriate device.
            x = batch['x'].to(device, dtype=torch.float32)
            y = batch['y'].to(device, dtype=torch.float32)
            x_len = batch['x_len'].to(device, dtype=torch.long)
            c = batch['c'].to(device, dtype=torch.long)
            c_len = batch['c_len'].to(device, dtype=torch.long)

            # print shpaes
            print(x.shape, y.shape, x_len.shape, c.shape, c_len.shape)
            
            optimizer.zero_grad()
            loss = model(x, y, x_len, c, c_len, sample_tsteps, num_samples, prime=False, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
            
            step += 1
            if step >= num_training_steps:
                break

    # Save the model or perform evaluation as needed.
