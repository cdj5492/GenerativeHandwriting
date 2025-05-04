import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import train_offset_normalization, valid_offset_normalization
from utils.constants import Global


class HandwritingDataset(Dataset):
    """Handwriting dataset."""

    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        """
        Args:
            data_path (string): Path to the data folder.
            split (string): train or valid
        """
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        strokes = np.load(data_path + 'strokes.npy', allow_pickle=True, encoding='bytes')
        with open(data_path + 'sentences.txt') as file:
            texts = file.read().splitlines()

        # list of length of each stroke in strokes
        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths)
        n_total = len(strokes)

        # Mask
        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        # Convert list of str into array of list of chars
        char_seqs = [list(char_seq) for char_seq in texts]
        # convert to numpy array, taking into account that it is non-homogeneous
        char_seqs = np.array(char_seqs, dtype=object)

        char_lens = [len(char_seq) for char_seq in char_seqs]
        max_char_len = np.max(char_lens)

        # char Mask
        mask_shape = (n_total, max_char_len)  # (6000,64)
        char_mask = np.zeros(mask_shape, dtype=np.float32)

        # Input text array
        inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
        inp_text[:, :] = ' '

        # Convert list of stroke(array) into ndarray of size(n_total, max_len, 3)
        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, (seq_len, text_len) in enumerate(zip(lengths, char_lens)):
            mask[i, :seq_len] = 1.
            data[i, :seq_len] = strokes[i]
            char_mask[i, :text_len] = 1.
            inp_text[i, :text_len] = char_seqs[i]

        # create vocab
        self.id_to_char, self.char_to_id = self.build_vocab(inp_text)
        self.vocab_size = len(self.id_to_char)

        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        inp_text = inp_text[idx_permute]
        char_mask = char_mask[idx_permute]

        if debug:
            data = data[:64]
            mask = mask[:64]
            inp_text = inp_text[:64]
            char_mask = char_mask[:64]

        n_train = int(0.9 * data.shape[0])
        self._data = data
        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.texts = inp_text[:n_train]
            self.char_mask = char_mask[:n_train]
            Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(
                self.dataset)

        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.texts = inp_text[n_train:]
            self.char_mask = char_mask[n_train:]
            self.dataset = valid_offset_normalization(
                Global.train_mean, Global.train_std, self.dataset)

    def __len__(self):
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        return np.array([self.id_to_char[id] for id in id_seq])

    def char_to_idx(self, char_seq):
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.int8)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text)
        unique_char = sorted(counter)
        vocab_size = len(unique_char)

        id_to_char = dict(zip(np.arange(vocab_size), unique_char))
        char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
        return id_to_char, char_to_id

    def __getitem__(self, idx):

        mask = torch.from_numpy(self.mask[idx])

        if self.text_req:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])

            target = torch.from_numpy(self.dataset[idx])
            text = torch.from_numpy(self.char_to_idx(self.texts[idx]))
            char_mask = torch.from_numpy(self.char_mask[idx])
            return (input_seq, target, mask, text, char_mask)
        elif self.data_aug:
            seq_len = len(mask.nonzero())
            start = 0
            end = self.max_seq_len

            if seq_len > self.max_seq_len:
                start = np.random.randint(0, high=seq_len - self.max_seq_len)
                end = start + self.max_seq_len

            stroke = self.dataset[idx, start:end, :]

            input_seq = torch.zeros(stroke.shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(stroke[:-1, :])

            target = torch.from_numpy(stroke)
            mask = mask[start:end]

            return (input_seq, target, mask)
        else:
            input_seq = torch.zeros(self.dataset[idx].shape, dtype=torch.float32)
            input_seq[1:, :] = torch.from_numpy(self.dataset[idx, :-1, :])
            target = torch.from_numpy(self.dataset[idx])
            return (input_seq, target, mask)

# --- Token Definitions ---
PAD_TOK = "<pad>"
START_SQRT_TOK = "\\sqrt{"
END_SQRT_TOK = "}"
START_FRAC_TOK = "\\frac{"
MID_FRAC_TOK = "}{"
END_FRAC_TOK = "}"
PI_TOK = "\\pi"
TIMES_TOK_STR = "\\times"
NEQ_TOK_STR = "\\neq"
LEQ_TOK_STR = "\\leq"
GEQ_TOK_STR = "\\geq"

GREEK_COMMANDS = [
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\theta", "\\lambda",
    "\\mu", "\\nu", "\\xi", "\\rho", "\\sigma", "\\tau", "\\phi", "\\chi",
    "\\psi", "\\omega"
]

SPECIAL_TOKENS = [PAD_TOK, START_SQRT_TOK, END_SQRT_TOK, START_FRAC_TOK, MID_FRAC_TOK, PI_TOK, TIMES_TOK_STR, NEQ_TOK_STR, LEQ_TOK_STR, GEQ_TOK_STR] + GREEK_COMMANDS

# --- Helper Functions for LaTeX Tokenization ---

def _find_matching_brace(text, start_index):
    open_brace_pos = -1
    search_idx = start_index
    while search_idx < len(text):
        if text[search_idx] == '{':
            open_brace_pos = search_idx
            break
        search_idx += 1
    if open_brace_pos == -1: return -1
    brace_level = 1
    i = open_brace_pos + 1
    while i < len(text):
        if text[i] == '{': brace_level += 1
        elif text[i] == '}':
            brace_level -= 1
            if brace_level == 0: return i
        i += 1
    return -1

def _tokenize_latex(text):
    # Expects a single string, already space-removed if desired
    # text = text.replace(" ", "") # Space removal now happens before calling this
    tokens = []
    i = 0
    while i < len(text):
        if text.startswith("\\sqrt{", i):
            cmd_len = len("\\sqrt{")
            match_brace_idx = _find_matching_brace(text, i + cmd_len - 1)
            if match_brace_idx != -1:
                tokens.append(START_SQRT_TOK)
                content_start = i + cmd_len
                content = text[content_start : match_brace_idx]
                tokens.extend(_tokenize_latex(content)) # Recursive call on content
                tokens.append(END_SQRT_TOK)
                i = match_brace_idx + 1
            else:
                tokens.append(text[i]); i += 1 # Malformed, treat as chars
        elif text.startswith("\\frac{", i):
            cmd_len = len("\\frac{")
            num_start_brace_pos = i + cmd_len - 1
            num_end_brace_pos = _find_matching_brace(text, num_start_brace_pos)
            if num_end_brace_pos != -1 and (num_end_brace_pos + 1) < len(text) and text[num_end_brace_pos + 1] == '{':
                den_start_brace_pos = num_end_brace_pos + 1
                den_end_brace_pos = _find_matching_brace(text, den_start_brace_pos)
                if den_end_brace_pos != -1:
                    tokens.append(START_FRAC_TOK)
                    num_content = text[num_start_brace_pos + 1 : num_end_brace_pos]
                    tokens.extend(_tokenize_latex(num_content))
                    tokens.append(MID_FRAC_TOK)
                    den_content = text[den_start_brace_pos + 1 : den_end_brace_pos]
                    tokens.extend(_tokenize_latex(den_content))
                    tokens.append(END_FRAC_TOK)
                    i = den_end_brace_pos + 1
                else:
                    tokens.append(text[i]); i += 1 # Malformed denominator
            else:
                tokens.append(text[i]); i += 1 # Malformed numerator
        elif text.startswith("\\pi", i):
            tokens.append(PI_TOK)
            i += len("\\pi")
        elif text[i] == '{':
            match_brace_idx = _find_matching_brace(text, i)
            if match_brace_idx != -1:
                # Assumes non-command braces are for grouping; tokenizes content
                content = text[i + 1 : match_brace_idx]
                tokens.extend(_tokenize_latex(content))
                i = match_brace_idx + 1
            else:
                tokens.append('{'); i += 1 # Unmatched brace
        else:
            # Append other characters (includes '}')
            tokens.append(text[i]); i += 1
    return tokens

class MathHandwritingDataset(Dataset):
    """Handwriting dataset with LaTeX tokenization."""

    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        # --- Load Raw Data ---
        strokes_path = data_path + 'strokes.npy'
        texts_path = data_path + 'sentences.txt'
        try:
            strokes = np.load(strokes_path, allow_pickle=True, encoding='latin1')
            with open(texts_path, 'r', encoding='utf-8') as file:
                raw_texts = file.read().splitlines()
            print(f"Loaded {len(strokes)} strokes and {len(raw_texts)} sentences.")
            if len(strokes) != len(raw_texts):
                 print(f"Warning: Stroke/sentence count mismatch ({len(strokes)}/{len(raw_texts)}). Trimming to shorter length.")
                 min_len = min(len(strokes), len(raw_texts))
                 strokes = strokes[:min_len]
                 raw_texts = raw_texts[:min_len]
        except FileNotFoundError as e:
            print(f"Error loading data: {e}.")
            raise e

        # --- Prepare Raw Text for Tokenization (Remove Spaces) ---
        # Create clean strings first
        cleaned_texts = [text.replace(" ", "") for text in raw_texts]
        # Also create list of character lists if needed for char_to_idx input later
        # char_seqs = [list(ct) for ct in cleaned_texts] # If char_to_idx strictly needs char list

        # --- Tokenize ONCE for Vocabulary Building ---
        print("Tokenizing text for vocabulary...")
        tokenized_for_vocab = [_tokenize_latex(text) for text in cleaned_texts]
        print("Tokenization for vocabulary complete.")

        # --- Build Vocabulary ---
        # Pass the token lists generated above
        self.idx_to_tok_map, self.tok_to_idx_map = self.build_vocab(tokenized_for_vocab)
        self.vocab_size = len(self.idx_to_tok_map)
        print(f"Vocabulary size: {self.vocab_size}")


        # --- Convert to Final Index Sequences using char_to_idx ---
        # The char_to_idx method will now handle tokenization internally.
        # We pass the cleaned *string* representation (or char list if strictly needed)
        # NOTE: This implies tokenizing *again* inside char_to_idx for each sentence.
        print("Converting sequences to indices using char_to_idx...")
        # Pass the list of characters derived from the cleaned text
        tok_indices = [self.char_to_idx(list(text)) for text in cleaned_texts]
        print("Index conversion complete.")


        # --- Pad Index Sequences ---
        pad_idx = self.tok_to_idx_map.get(PAD_TOK, 0)
        if PAD_TOK not in self.tok_to_idx_map: print("Warning: PAD_TOK not in vocabulary!")

        tok_lens = [len(seq) for seq in tok_indices] # Lengths are based on the *output* of char_to_idx
        max_tok_len = np.max(tok_lens) if tok_lens else 0
        print(f"Maximum token sequence length (after tokenization): {max_tok_len}")
        n_total = len(strokes) # Use length after potential trimming

        padded_tok_indices = np.full((n_total, max_tok_len), pad_idx, dtype=np.int8)
        for i, seq in enumerate(tok_indices):
            current_len = len(seq)
            if current_len > max_tok_len:
                 padded_tok_indices[i, :] = seq[:max_tok_len] # Truncate if somehow longer
            else:
                 padded_tok_indices[i, :current_len] = seq

        tok_mask = np.zeros((n_total, max_tok_len), dtype=np.float32)
        for i, length in enumerate(tok_lens):
            actual_len = min(length, max_tok_len)
            tok_mask[i, :actual_len] = 1.

        # --- Prepare Stroke Data (Padding and Masking) ---
        stroke_lens = [len(stroke) for stroke in strokes]
        max_stroke_len = np.max(stroke_lens) if stroke_lens else 0
        print(f"Maximum stroke sequence length: {max_stroke_len}")

        stroke_mask = np.zeros((n_total, max_stroke_len), dtype=np.float32)
        stroke_data = np.zeros((n_total, max_stroke_len, 3), dtype=np.float32)

        for i, (stroke_len, stroke) in enumerate(zip(stroke_lens, strokes)):
            actual_len = min(stroke_len, max_stroke_len)
            if actual_len > 0:
                 stroke_mask[i, :actual_len] = 1.
                 stroke_np = np.array(stroke, dtype=np.float32) if not isinstance(stroke, np.ndarray) else stroke.astype(np.float32)
                 stroke_data[i, :actual_len] = stroke_np[:actual_len]

        # --- Shuffle Data ---
        idx_permute = np.random.permutation(n_total)
        stroke_data = stroke_data[idx_permute]
        stroke_mask = stroke_mask[idx_permute]
        self.token_sequences = padded_tok_indices[idx_permute] # Store final padded indices
        self.token_mask = tok_mask[idx_permute]

        # --- Debug Subset ---
        if debug:
            n_debug = min(64, n_total)
            stroke_data = stroke_data[:n_debug]
            stroke_mask = stroke_mask[:n_debug]
            self.token_sequences = self.token_sequences[:n_debug]
            self.token_mask = self.token_mask[:n_debug]
            print(f"Using debug subset of size {n_debug}")

        # --- Split Data ---
        n_samples = stroke_data.shape[0]
        n_train = int(0.9 * n_samples)
        if n_samples == 0:
             print("Warning: No data samples after processing.")
             # Init empty attributes
             self.dataset = np.empty((0, max_stroke_len, 3), dtype=np.float32); self.mask = np.empty((0, max_stroke_len), dtype=np.float32)
             self.texts = np.empty((0, max_tok_len), dtype=np.int8); self.char_mask = np.empty((0, max_tok_len), dtype=np.float32)
             return

        print(f"Total samples: {n_samples}, Training samples: {n_train}, Validation samples: {n_samples - n_train}")

        if split == 'train':
            self.dataset = stroke_data[:n_train]; self.mask = stroke_mask[:n_train]
            self.texts = self.token_sequences[:n_train]; self.char_mask = self.token_mask[:n_train]
            if self.dataset.size > 0:
                Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(self.dataset)
                print(f"Train data normalized. Mean: {Global.train_mean}, Std: {Global.train_std}")
            else: print("Skipping training normalization (no data).")
        elif split == 'valid':
            if n_train >= n_samples:
                 print("Warning: No validation samples after split."); n_val = 0
                 self.dataset = np.empty((0, max_stroke_len, 3), dtype=np.float32); self.mask = np.empty((0, max_stroke_len), dtype=np.float32)
                 self.texts = np.empty((0, max_tok_len), dtype=np.int8); self.char_mask = np.empty((0, max_tok_len), dtype=np.float32)
            else:
                self.dataset = stroke_data[n_train:]; self.mask = stroke_mask[n_train:]
                self.texts = self.token_sequences[n_train:]; self.char_mask = self.token_mask[n_train:]
                if self.dataset.size > 0:
                    self.dataset = valid_offset_normalization(Global.train_mean, Global.train_std, self.dataset)
                    print("Validation data normalized using training stats.")
                else: print("Skipping validation normalization (no data).")
        else:
             raise ValueError(f"Invalid split: {split}. Choose 'train' or 'valid'.")


    def build_vocab(self, tokenized_texts_list):
        """Builds vocabulary from a list of PRE-TOKENIZED sequences."""
        counter = Counter()
        for tokens in tokenized_texts_list:
            counter.update(tokens)
        # Build vocab based on these tokens
        unique_tokens = [PAD_TOK]
        unique_tokens.extend(sorted(list(set(tok for tok in SPECIAL_TOKENS if tok != PAD_TOK))))
        data_tokens = sorted([tok for tok in counter if tok not in unique_tokens])
        unique_tokens.extend(data_tokens)
        final_unique_tokens = []; seen = set()
        for tok in unique_tokens:
            if tok not in seen: final_unique_tokens.append(tok); seen.add(tok)
        idx_to_tok = {i: tok for i, tok in enumerate(final_unique_tokens)}
        tok_to_idx = {tok: i for i, tok in idx_to_tok.items()}
        for special in SPECIAL_TOKENS:
             if special not in tok_to_idx: print(f"WARNING: Special token '{special}' missing!")
        return idx_to_tok, tok_to_idx

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        """Converts a sequence of token indices back to token strings."""
        # Name kept as idx_to_char, but operates on tokens
        return np.array([self.idx_to_tok_map.get(int(idx), "<unk>") for idx in id_seq])

    def char_to_idx(self, char_seq):
        """
        Converts a list of characters to a sequence of token indices.
        Performs LaTeX tokenization internally.
        Input: List of characters (e.g., ['3', '(', '5', ...])
        Output: numpy array of int8 token indices.
        """
        # 1. Join characters into a single string (assume no spaces needed)
        text = "".join(char_seq)
        # 2. Tokenize the string using the LaTeX tokenizer
        tokens = _tokenize_latex(text)
        # 3. Map the resulting tokens to indices
        pad_idx = self.tok_to_idx_map.get(PAD_TOK, 0) # Fallback for unknown?
        # Map unknown tokens to -1 or pad_idx. Using -1.
        indices = np.array([self.tok_to_idx_map.get(tok, -1) for tok in tokens], dtype=np.int8)
        # Check for -1s which indicate tokens not in vocab
        if -1 in indices:
            unknown_tokens = [tokens[i] for i, idx in enumerate(indices) if idx == -1]
            print(f"Warning: Unknown tokens encountered in char_to_idx: {set(unknown_tokens)}")
        return indices

    def __getitem__(self, idx):
        """Gets the idx-th sample from the dataset."""
        if idx >= len(self): raise IndexError("Index out of range")

        stroke_mask = torch.from_numpy(self.mask[idx]).float()
        stroke_data = self.dataset[idx] # Already normalized numpy array
        stroke_data_tensor = torch.from_numpy(stroke_data).float()

        if self.text_req:
            input_seq = torch.zeros_like(stroke_data_tensor)
            if stroke_data_tensor.shape[0] > 1:
                 input_seq[1:, :] = stroke_data_tensor[:-1, :]
            target = stroke_data_tensor
            # Return the pre-computed token indices and mask
            text_indices = torch.from_numpy(self.texts[idx]).long()
            token_mask = torch.from_numpy(self.char_mask[idx]).float()
            return (input_seq, target, stroke_mask, text_indices, token_mask)
        elif self.data_aug:
            # Data augmentation logic (as before)
            seq_len = int(stroke_mask.sum().item())
            start = 0; data_len = stroke_data_tensor.shape[0]; end = min(seq_len, data_len)
            if seq_len > self.max_seq_len:
                high = max(1, seq_len - self.max_seq_len + 1)
                start = np.random.randint(0, high=high)
                end = start + self.max_seq_len
            start = min(start, data_len); end = min(end, data_len)
            if start >= end and data_len > 0: start = max(0, data_len - 1); end = data_len
            elif start >= end and data_len == 0: start, end = 0, 0
            cropped_stroke = stroke_data_tensor[start:end, :]; cropped_mask = stroke_mask[start:end]
            input_seq = torch.zeros_like(cropped_stroke)
            if cropped_stroke.shape[0] > 1: input_seq[1:, :] = cropped_stroke[:-1, :]
            target = cropped_stroke
            return (input_seq, target, cropped_mask)
        else:
            # Standard return (no text, no aug)
            input_seq = torch.zeros_like(stroke_data_tensor)
            if stroke_data_tensor.shape[0] > 1: input_seq[1:, :] = stroke_data_tensor[:-1, :]
            target = stroke_data_tensor
            return (input_seq, target, stroke_mask)