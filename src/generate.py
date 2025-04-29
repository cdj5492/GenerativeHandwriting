import torch
import numpy as np
import argparse
import matplotlib
from pathlib import Path
import os
import matplotlib.pyplot as plt
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset, MathHandwritingDataset
from utils.data_utils import data_denormalization, data_normalization
from models import HandWritingPredictionNet, HandWritingSynthesisNet
from transformer_handwriting_models import (
    HandWritingPredictionNet as TransformerHandWritingPredictionNet,
    HandWritingSynthesisNet as TransformerHandWritingSynthesisNet,
)


def argparser():

    parser = argparse.ArgumentParser(description="PyTorch Handwriting Synthesis Model")
    parser.add_argument("--model", type=str, default="synthesis")
    parser.add_argument(
        "--model_path",
        type=Path,
        default="./results/synthesis/best_model_synthesis_3.pt",
    )
    parser.add_argument("--save_path", type=Path, default="./results/")
    parser.add_argument("--seq_len", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bias", type=float, default=10.0, help="bias")
    parser.add_argument("--char_seq", type=str, default="This is real handwriting")
    parser.add_argument("--text_req", action="store_true")
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--is_map", action="store_true")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--file_path", type=str, help="./app/")
    args = parser.parse_args()

    return args

# sample function
def sample(preds, temperature=1.0):
    # helper function to sample an element from a probability distribution
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_unconditional_seq(model_path, seq_len, device, bias, style, prime, model_arch='transformer'):

    if model_arch == 'transformer':
        model = TransformerHandWritingPredictionNet()
    else:
        model = HandWritingPredictionNet()
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # initial input
    inp = torch.zeros(1, 1, 3)
    inp = inp.to(device)

    batch_size = 1

    print("Generating sequence....")

    if model_arch == 'transformer':
        gen_seq = model.generate(prime, seq_len, bias)
    else:
        initial_hidden = model.init_hidden(batch_size, device)
        gen_seq = model.generate(inp, initial_hidden, seq_len, bias, style, prime)

    return gen_seq

# --- Utility function for padding sequences in a batch ---
def pad_sequences(sequences, max_len, pad_value):
    """Pads a list of sequences to the same length."""
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length > 0:
            padded[i, :length] = seq[:max_len] # Truncate if longer than max_len
    return padded

def create_mask(padded_sequences, pad_value):
    """Creates a mask for padded sequences (1 for real tokens, 0 for padding)."""
    mask = (padded_sequences != pad_value).astype(np.float32)
    return mask
# ---


def generate_conditional_sequence(
    model_path: str,
    char_seq: str,            # The input text sequence (e.g., LaTeX string)
    device: torch.device,
    dataset: HandwritingDataset | MathHandwritingDataset, # Pass an instance of the dataset
    bias: float,
    prime: bool,
    prime_seq: torch.Tensor,  # Assuming this is already a tensor [batch, seq, 3]
    real_text: str,           # The priming text sequence (e.g., LaTeX string)
    is_map: bool,
    batch_size: int = 1,
    model_arch: str = 'transformer',
):
    """
    Generates a handwriting sequence conditioned on input text using a pre-trained model.

    Args:
        model_path (str): Path to the saved model state dictionary.
        char_seq (str): The target text sequence to generate handwriting for.
        device (torch.device): The device (CPU or GPU) to run generation on.
        dataset (HandwritingDataset): An initialized instance of the HandwritingDataset,
                                       which contains the tokenizer methods and vocabulary.
        bias (float): Sampling bias for generation.
        prime (bool): Whether to use priming (starting the sequence with given strokes).
        prime_seq (torch.Tensor): The initial stroke sequence tensor for priming.
                                  Shape: [batch_size, prime_len, 3].
        real_text (str): The text corresponding to the prime_seq, used for attention priming.
        is_map (bool): Flag used by the model's generate method (e.g., for MAP decoding).
        batch_size (int): Batch size for generation (usually 1 for inference).

    Returns:
        tuple: (generated_sequence_tensor, attention_weights_phi)
               - generated_sequence_tensor (torch.Tensor): The generated stroke sequence.
               - attention_weights_phi (np.ndarray or list): Attention weights (phi).
    """

    # Use vocab size from the dataset instance
    if model_arch == 'transformer':
        model = TransformerHandWritingSynthesisNet(window_size=dataset.vocab_size)
    else:
        model = HandWritingSynthesisNet(window_size=dataset.vocab_size)
    print(f"Using vocab size from dataset: {dataset.vocab_size}")

    # Load the best model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

    model = model.to(device)
    model.eval()

    # --- Prepare Priming Data (if used) ---
    prime_text_tensor = None
    prime_mask = None
    inp = None # Initialize inp

    if prime:
        if prime_seq is None or real_text is None:
            raise ValueError("prime_seq and real_text must be provided when prime=True")

        inp = prime_seq.to(device) # Ensure prime_seq is on the correct device

        # Tokenize the priming text using the dataset's method
        # The result is a list of token IDs
        token_id_list = dataset.char_to_idx(real_text).tolist()
        if not token_id_list:
             print("Warning: Priming text resulted in empty token list.")
             # Handle this case: maybe raise error, or create a dummy tensor?
             # Creating a tensor with a single <PAD> or <SOS> might be an option
             # For now, let's create a minimal tensor based on expected shape later
             token_id_list = [dataset.pad_token_id] # Use padding token ID


        # Create batch and convert to tensor (use LongTensor for token IDs)
        prime_text_np = np.array([token_id_list for _ in range(batch_size)], dtype=np.int32)
        prime_text_tensor = torch.from_numpy(prime_text_np).long().to(device)

        # Create mask for priming text (assuming no padding within real_text itself)
        prime_mask = torch.ones(prime_text_tensor.shape, dtype=torch.float32).to(device) # Use float for mask

        print(f"Priming with {inp.shape[1]} stroke steps and text: '{real_text}'")

    else:
        # Initial input for non-primed generation (start token equivalent for strokes)
        inp = torch.zeros(batch_size, 1, 3, dtype=torch.float32).to(device) # Use float for strokes

    # --- Prepare Target Text Data ---
    # Tokenize the target text sequence using the dataset's method
    # Note: Removed adding "  " - rely on tokenizer/model to handle EOS if needed
    # If EOS is required, add it explicitly: dataset.char_to_idx(char_seq + dataset.id_to_token[dataset.eos_token_id])
    target_token_id_list = dataset.char_to_idx(char_seq).tolist()
    if not target_token_id_list:
        print(f"Warning: Input text '{char_seq}' resulted in empty token list. Using a PAD token.")
        target_token_id_list = [dataset.pad_token_id] # Use padding token ID


    # Create batch and convert to tensor (use LongTensor for token IDs)
    text_np = np.array([target_token_id_list for _ in range(batch_size)], dtype=np.int32)
    text_tensor = torch.from_numpy(text_np).long().to(device)

    # Create mask for target text (assuming no padding here, change if sequences are padded)
    text_mask = torch.ones(text_tensor.shape, dtype=torch.float32).to(device) # Use float for mask

    # --- Generate Sequence ---
    print(f"Generating sequence for text: '{char_seq}'")

    if model_arch == 'transformer':
        with torch.no_grad():
            pass
            # TODO: Implement the generation logic for transformer models
            # gen_seq = model.generate(
            #     prime=prime,
            #     text=text_tensor,
            #     seq_len=seq_len,
            #     bias=bias,
            #     sample_fn=sample,
            # )
    else:
        # --- Initialize Model State ---
        hidden, window_vector, kappa = model.init_hidden(batch_size, device)
        with torch.no_grad(): # Ensure gradients are not computed during generation
            gen_seq = model.generate(
                inp=inp,                    # Initial stroke input (prime_seq or zeros)
                text=text_tensor,           # Target text token IDs
                text_mask=text_mask,        # Mask for target text
                prime_text=prime_text_tensor, # Priming text token IDs (or None)
                prime_mask=prime_mask,      # Mask for priming text (or None)
                hidden=hidden,              # Initial hidden state
                window_vector=window_vector,# Initial window vector for attention
                kappa=kappa,                # Initial kappa for attention
                bias=bias,                  # Sampling bias
                is_map=is_map,              # MAP flag
                prime=prime,                # Priming flag passed to model's generate
            )

    # --- Decode Input Text for Verification ---
    # Use the dataset's idx_to_char method which returns a list of token strings
    input_tokens_list = dataset.idx_to_char(text_tensor[0].cpu().numpy())
    # Join the tokens into a readable string (simple join, might need spaces)
    decoded_input_text = "".join(input_tokens_list)

    # The original length calculation might be misleading if mask isn't just padding
    # Print the decoded text based on the list of tokens retrieved
    print(f"Input token IDs shape: {text_tensor[0].shape[0]}")
    print(f"Decoded input text: '{decoded_input_text}'")


    # --- Extract Attention Weights (if applicable) ---
    phi = []
    if hasattr(model, '_phi') and model._phi is not None: # Check if model stored phi
        # Check if _phi is a list and not empty
        if isinstance(model._phi, list) and model._phi:
            try:
                phi_tensor = torch.cat(model._phi, dim=1) # Concatenate along sequence dim
                 # Check dimensions before transpose, handle potential batch > 1
                if phi_tensor.dim() >= 3 and phi_tensor.shape[0] > 0: # Batch, Seq, Features
                     phi = phi_tensor[0].cpu().numpy().T # Take first batch item, transpose
                elif phi_tensor.dim() == 2: # If already concatenated differently (Seq, Feat)
                     phi = phi_tensor.cpu().numpy().T
                else:
                     print("Warning: Unexpected shape for concatenated phi tensor.")
                     phi = phi_tensor.cpu().numpy() # Return as is

            except Exception as e:
                print(f"Warning: Could not process attention weights (phi): {e}")
                # Optionally try returning the raw list if concatenation fails
                # phi = [p[0].cpu().numpy().T for p in model._phi if p.shape[0]>0]
        else:
            print("Warning: Model attribute '_phi' found but is not a non-empty list.")
    elif is_map:
         print("Warning: is_map is True, but attention weights (phi) not found in model._phi")


    return gen_seq, phi


if __name__ == "__main__":

    args = argparser()
    if not args.save_path.exists():
        args.save_path.mkdir(parents=True, exist_ok=True)

    # fix random seed
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = args.model

    train_dataset = HandwritingDataset(
        args.data_path, split="train", text_req=args.text_req
    )

    if args.prime and args.file_path:
        style = np.load(
            args.file_path + "style.npy", allow_pickle=True, encoding="bytes"
        ).astype(np.float32)
        with open(args.file_path + "inpText.txt") as file:
            texts = file.read().splitlines()
        real_text = texts[0]
        # plot the sequence
        plot_stroke(style, save_name=args.save_path / "style.png")
        print(real_text)
        mean, std, _ = data_normalization(style)
        style = torch.from_numpy(style).unsqueeze(0).to(device)
        print(style.shape)
        ytext = real_text + " " + args.char_seq + "  "
    elif args.prime:
        strokes = np.load(
            args.data_path + "strokes.npy", allow_pickle=True, encoding="bytes"
        )
        with open(args.data_path + "sentences.txt") as file:
            texts = file.read().splitlines()
        idx = np.random.randint(0, len(strokes))
        print("Prime style index: ", idx)
        real_text = texts[idx]
        style = strokes[idx]
        # plot the sequence
        plot_stroke(style, save_name=args.save_path / ("style_" + str(idx) + ".png"))
        print(real_text)
        mean, std, _ = data_normalization(style)
        style = np.array([style for i in range(args.batch_size)])
        style = torch.from_numpy(style).to(device)
        print(style.shape)
        ytext = real_text + " " + args.char_seq + "  "
    else:
        idx = -1
        real_text = ""
        style = None
        ytext = args.char_seq + "  "

    if model == "prediction":
        gen_seq = generate_unconditional_seq(
            model_path, args.seq_len, device, args.bias, style=style, prime=args.prime
        )
    elif model == "synthesis":
        gen_seq, phi = generate_conditional_sequence(
            model_path,
            args.char_seq,
            device,
            train_dataset.char_to_id,
            train_dataset.idx_to_char,
            args.bias,
            args.prime,
            style,
            real_text,
            args.is_map,
            args.batch_size,
        )
        if args.is_map:
            plt.imshow(phi, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.xlabel("time steps")
            plt.yticks(np.arange(phi.shape[0]), list(ytext), rotation="horizontal")
            plt.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
            plt.savefig("heat_map.png")
            plt.close()

    # denormalize the generated offsets using train set mean and std
    # if args.prime:
    #     print("data denormalization...")
    #     gen_seq = data_denormalization(mean, std, gen_seq)
    # else:
    gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    # plot the sequence
    for i in range(args.batch_size):
        plot_stroke(
            gen_seq[i], save_name=args.save_path / ("gen_seq_" + str(i) + ".png")
        )