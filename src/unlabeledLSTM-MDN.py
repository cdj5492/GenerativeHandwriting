import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from math import sqrt, pi
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
from matplotlib.path import Path
import matplotlib.patches as patches
import random

class HandwritingMDN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, num_mixtures=20):
        """
        Mixture Density Network for handwriting generation
        
        Args:
            input_dim: dimension of input (dx, dy, pen_state)
            hidden_dim: dimension of LSTM hidden state
            num_layers: number of LSTM layers
            num_mixtures: number of Gaussian mixtures for coordinate modeling
        """
        super(HandwritingMDN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        
        # LSTM to model the sequence
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output layers
        # For each mixture: pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy (correlation)
        self.mdn_layer = nn.Linear(hidden_dim, num_mixtures * 6)
        
        # Separate layer for pen state prediction
        self.pen_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with sensible defaults for MDN outputs"""
        # Initialize the MDN layer with small weights
        nn.init.normal_(self.mdn_layer.weight, 0, 0.01)
        
        # Initialize biases for sigma to be slightly positive to prevent zero variance
        if self.mdn_layer.bias is not None:
            # Get bias indices for sigma_x and sigma_y
            sigma_idx = np.array([i * 6 + j for i in range(self.num_mixtures) for j in [3, 4]])
            # Set those biases to log(1) = 0
            self.mdn_layer.bias.data[sigma_idx] = 0
            
            # Initialize correlation rho to be small
            rho_idx = np.array([i * 6 + 5 for i in range(self.num_mixtures)])
            self.mdn_layer.bias.data[rho_idx] = 0
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Initial hidden state (optional)
            
        Returns:
            pi: Mixture weights (batch_size, seq_len, num_mixtures)
            mu: Means (batch_size, seq_len, num_mixtures, 2)
            sigma: Standard deviations (batch_size, seq_len, num_mixtures, 2)
            rho: Correlation between x and y (batch_size, seq_len, num_mixtures)
            pen_prob: Probability of pen-up (batch_size, seq_len, 1)
            hidden: LSTM hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply MDN layer
        mdn_out = self.mdn_layer(lstm_out)
        
        # Reshape MDN outputs
        batch_size, seq_len, _ = mdn_out.size()
        mdn_out = mdn_out.view(batch_size, seq_len, self.num_mixtures, 6)
        
        # Extract MDN parameters with appropriate activations
        pi = F.softmax(mdn_out[:, :, :, 0], dim=2)  # Mixing coefficients
        mu_x = mdn_out[:, :, :, 1]  # Mean x (no activation, can be any real number)
        mu_y = mdn_out[:, :, :, 2]  # Mean y (no activation, can be any real number)
        sigma_x = torch.exp(mdn_out[:, :, :, 3])  # Standard deviation x (must be positive)
        sigma_y = torch.exp(mdn_out[:, :, :, 4])  # Standard deviation y (must be positive)
        rho = torch.tanh(mdn_out[:, :, :, 5])  # Correlation between x and y (between -1 and 1)
        
        # Combine mu and sigma for easier handling
        mu = torch.stack([mu_x, mu_y], dim=3)  # Shape: (batch_size, seq_len, num_mixtures, 2)
        sigma = torch.stack([sigma_x, sigma_y], dim=3)  # Shape: (batch_size, seq_len, num_mixtures, 2)
        
        # Apply pen state layer
        pen_logits = self.pen_layer(lstm_out)
        pen_prob = torch.sigmoid(pen_logits)  # Probability of pen-up
        
        return pi, mu, sigma, rho, pen_prob, hidden
    
    def sample(self, seq_len, temperature=1.0, initial_input=None, device='cpu'):
        """
        Generate a handwriting sample
        
        Args:
            seq_len: Length of sequence to generate
            temperature: Temperature for sampling (higher = more random)
            initial_input: Optional initial input (shape should be [seq_len, 3])
            device: Device to use
            
        Returns:
            Generated sequence of (dx, dy, pen_state)
        """
        self.eval()
        with torch.no_grad():
            # Initialize hidden state
            hidden = None
            generated_sequence = []
            
            # Start with zero input if none provided
            if initial_input is None:
                # Use a zero tensor as starting point
                current_input = torch.zeros(1, 1, self.input_dim, device=device)
            else:
                # Process the initial input sequence first to get a proper hidden state
                # Make sure initial_input has batch dimension
                if initial_input.dim() == 2:
                    initial_input = initial_input.unsqueeze(0)  # Add batch dimension
                
                # Get hidden state from initial input
                _, hidden = self.lstm(initial_input, None)
                
                # Start with the last point from initial input
                current_input = initial_input[:, -1:, :]
            
            # Generate the sequence point by point
            for i in range(seq_len):
                # Forward pass
                lstm_out, hidden = self.lstm(current_input, hidden)
                
                # Apply MDN layer
                mdn_out = self.mdn_layer(lstm_out)
                
                # Reshape MDN outputs
                batch_size, seq_len, _ = mdn_out.size()
                mdn_out = mdn_out.view(batch_size, seq_len, self.num_mixtures, 6)
                
                # Extract parameters
                pi = F.softmax(mdn_out[:, :, :, 0] / temperature, dim=2)
                mu_x = mdn_out[:, :, :, 1]
                mu_y = mdn_out[:, :, :, 2]
                sigma_x = torch.exp(mdn_out[:, :, :, 3]) * temperature
                sigma_y = torch.exp(mdn_out[:, :, :, 4]) * temperature
                rho = torch.tanh(mdn_out[:, :, :, 5])
                
                # Get pen state
                pen_logits = self.pen_layer(lstm_out)
                pen_prob = torch.sigmoid(pen_logits)
                
                # Sample from mixture
                # 1. Choose mixture component
                
                k = torch.multinomial(pi[0, 0], 1)[0]
                
                # 2. Sample from bivariate Gaussian
                mean_x = mu_x[0, 0, k]
                mean_y = mu_y[0, 0, k]
                std_x = sigma_x[0, 0, k]
                std_y = sigma_y[0, 0, k]
                correlation = rho[0, 0, k]
                
                # Sample with correlation
                z1 = torch.randn(1, device=device)
                z2 = torch.randn(1, device=device)
                
                dx = mean_x + std_x * z1
                dy = mean_y + std_y * (z1 * correlation + z2 * torch.sqrt(1 - correlation**2))
                
                # Sample pen state (Bernoulli)
                pen_state = torch.bernoulli(pen_prob[0, 0])
                
                # Create point and add to sequence
                point = torch.tensor([dx.item(), dy.item(), pen_state.item()], device=device)
                generated_sequence.append(point)
                
                # Update current input
                current_input = point.view(1, 1, -1)
            
            return torch.stack(generated_sequence)

def mdn_loss_fn(pi, mu, sigma, rho, pen_prob, target):
    """
    Compute the MDN loss function
    
    Args:
        pi: Mixture weights (batch_size, seq_len, num_mixtures)
        mu: Means (batch_size, seq_len, num_mixtures, 2)
        sigma: Standard deviations (batch_size, seq_len, num_mixtures, 2)
        rho: Correlation between x and y (batch_size, seq_len, num_mixtures)
        pen_prob: Probability of pen-up (batch_size, seq_len, 1)
        target: Target data (batch_size, seq_len, 3) - [dx, dy, pen_state]
        
    Returns:
        loss: Combined negative log likelihood
    """
    # Extract coordinate and pen targets
    coord_target = target[:, :, :2]  # dx, dy
    pen_target = target[:, :, 2:3]   # pen state
    
    # Expand coordinate targets for mixture components
    batch_size, seq_len = coord_target.size(0), coord_target.size(1)
    num_mixtures = pi.size(2)
    
    # Reshape targets for broadcasting: (batch, seq, 1, 2) -> (batch, seq, mixtures, 2)
    coord_target = coord_target.unsqueeze(2).expand(-1, -1, num_mixtures, -1)
    
    # Calculate the components of the bivariate Gaussian PDF
    x_target = coord_target[:, :, :, 0]
    y_target = coord_target[:, :, :, 1]
    
    mu_x = mu[:, :, :, 0]
    mu_y = mu[:, :, :, 1]
    
    sigma_x = sigma[:, :, :, 0]
    sigma_y = sigma[:, :, :, 1]
    
    # Add epsilon to prevent numerical issues
    epsilon = 1e-6
    sigma_x = torch.clamp(sigma_x, min=epsilon)
    sigma_y = torch.clamp(sigma_y, min=epsilon)
    
    # Calculate bivariate Gaussian PDF
    # Compute normalization constant
    z_x = (x_target - mu_x) / sigma_x
    z_y = (y_target - mu_y) / sigma_y
    
    # Account for correlation
    rho = torch.clamp(rho, -1 + epsilon, 1 - epsilon)  # Ensure |rho| < 1
    
    # Gaussian PDF formula terms
    term1 = -torch.log(2 * pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2) + epsilon)
    term2 = -0.5 / (1 - rho**2) * (z_x**2 + z_y**2 - 2 * rho * z_x * z_y)
    
    # Full PDF
    gaussian_log_probs = term1 + term2
    
    # Weight by the mixture coefficients
    weighted_log_probs = gaussian_log_probs + torch.log(pi + epsilon)
    
    # Log-sum-exp trick for numerical stability
    max_log_probs = torch.max(weighted_log_probs, dim=2, keepdim=True)[0]
    coord_log_likelihood = max_log_probs + torch.log(
        torch.sum(torch.exp(weighted_log_probs - max_log_probs), dim=2, keepdim=True) + epsilon
    )
    
    # Bernoulli loss for pen state
    pen_log_likelihood = torch.log(
        pen_target * pen_prob + (1 - pen_target) * (1 - pen_prob) + epsilon
    )
    
    # Combined loss (negative log-likelihood)
    total_log_likelihood = coord_log_likelihood + pen_log_likelihood
    loss = -torch.mean(total_log_likelihood)
    
    return loss

def train_model(model, data_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train the handwriting model
    
    Args:
        model: HandwritingMDN model
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use
        
    Returns:
        trained model and loss history
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in data_loader:
            batch = batch.to(device)
            
            # Split into input and target (shifted by 1)
            x = batch[:, :-1, :]
            y = batch[:, 1:, :]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pi, mu, sigma, rho, pen_prob, _ = model(x)
            
            # Compute loss
            loss = mdn_loss_fn(pi, mu, sigma, rho, pen_prob, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model, loss_history


# Import our custom modules (assuming they're saved in separate files)
from util.createUnlabeledTrainingData import generate_training_data
from util.visualize import visualize_sequence_global, visualize_sequence_delta
# from improved_handwriting_model import HandwritingMDN, train_model

# If you're running this in a single script, you'd have the classes defined above

def create_dataset(data_dir, max_files=None):
    """
    Creates a dataset from XML files in the specified directory
    
    Args:
        data_dir: Directory containing XML files
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        List of tensor sequences
    """
    file_list = glob("lineStrokes-all/lineStrokes/**/*.xml", recursive=True)
    if max_files:
        file_list = file_list[:max_files]
    
    print(f"Found {len(file_list)} XML files")
    sequences = generate_training_data(file_list)
    print(f"Generated {len(sequences)} sequences")
    
    return sequences

class HandwritingDataset(data.Dataset):
    """Dataset for handwriting sequences"""
    
    def __init__(self, sequences, seq_length=100):
        self.sequences = sequences
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # If sequence is shorter than seq_length, pad with zeros
        # if len(sequence) < self.seq_length:
        #     padding = torch.zeros(self.seq_length - len(sequence), 3)
        #     sequence = torch.cat([sequence, padding], dim=0)
        
        # If sequence is longer, take a random slice
        if len(sequence) > self.seq_length:
            start_idx = random.randint(0, len(sequence) - self.seq_length)
            sequence = sequence[start_idx:start_idx + self.seq_length]
            
        return sequence

def collate_variable_length(batch):
    """Custom collate function for variable length sequences"""
    # Find the minimum length that accommodates all sequences
    lengths = [x.size(0) for x in batch]
    max_len = max(lengths)
    
    # Pad all sequences to max_len
    padded_batch = []
    for seq in batch:
        if seq.size(0) < max_len:
            padding = torch.zeros(max_len - seq.size(0), 3)
            seq = torch.cat([seq, padding], dim=0)
        padded_batch.append(seq)
    
    return torch.stack(padded_batch)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    
    # Parameters
    data_dir = "lineStrokes-all/lineStrokes"
    max_files = 2000  # Set to None to use all files
    batch_size = 32
    seq_length = 1000
    num_epochs = 50
    learning_rate = 0.005
    
    # Create dataset
    print("Creating dataset...")
    sequences = create_dataset(data_dir, max_files)
    dataset = HandwritingDataset(sequences, seq_length)

    # visualize one sample from the dataset
    # visualize_sequence_delta(dataset[0], title="Sample from Dataset")
    
    # Create data loader
    data_loader = data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_variable_length
    )
    
    # Create model
    print("Creating model...")
    model = HandwritingMDN(
        input_dim=3,
        hidden_dim=256,
        num_layers=3,
        num_mixtures=20
    ).to(device)
    
    # Generate a sample before training
    print("Generating sample before training...")
    model.eval()
    with torch.no_grad():
        # Use a real sequence as seed (properly formatted)
        if len(sequences) > 0:
            # Take a short segment as seed
            seed_sequence = sequences[0][:5].clone().to(device)
            
            try:
                # Generate initial sample using seed
                initial_sample = model.sample(
                    seq_len=200,
                    temperature=1.0,
                    initial_input=seed_sequence,
                    device=device
                )
            except Exception as e:
                print(f"Error generating initial sample with seed: {e}")
                print("Generating sample without seed instead")
                initial_sample = model.sample(
                    seq_len=200,
                    temperature=1.0,
                    initial_input=None,
                    device=device
                )
        else:
            # No sequences available, generate without seed
            print("No sequences available, generating sample without seed")
            initial_sample = model.sample(
                seq_len=200,
                temperature=1.0,
                initial_input=None,
                device=device
            )
        
        # Visualize and save
        visualize_sequence_delta(
            initial_sample, 
            title="Handwriting Sample Before Training",
            save_path="plots/handwriting_before_training.png"
        )
    
    # Train model
    print("Training model...")
    model, loss_history = train_model(
        model,
        data_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/training_loss.png")
    
    # Generate a sample after training
    print("Generating sample after training...")
    model.eval()
    with torch.no_grad():
        # Try to use the same seed for comparison
        if len(sequences) > 0:
            seed_sequence = sequences[0][:5].clone().to(device)
            
            # Generate samples with different temperatures
            for temp in [0.05, 0.5, 1.0, 1.5]:
                trained_sample = model.sample(
                    seq_len=200,
                    temperature=temp,
                    initial_input=seed_sequence,
                    device=device
                )
                
                # Visualize and save
                # visualize_sequence_global(initial_sample, title=f"Handwriting Sample After Training (Temp={temp})")
                visualize_sequence_delta(
                    trained_sample, 
                    title=f"Handwriting Sample After Training (Temp={temp})",
                    save_path=f"plots/handwriting_after_training_temp{temp}.png"
                )
        else:
            # Generate without seed
            for temp in [0.05, 0.5, 1.0, 1.5]:
                trained_sample = model.sample(
                    seq_len=200,
                    temperature=temp,
                    initial_input=None,
                    device=device
                )
                
                visualize_sequence_delta(
                    trained_sample, 
                    title=f"Handwriting Sample After Training (Temp={temp})",
                    save_path=f"handwriting_after_training_temp{temp}.png"
                )
    
    # Save model
    torch.save(model.state_dict(), "models/handwriting_model.pt")
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()