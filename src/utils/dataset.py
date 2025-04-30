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
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.int64)

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

# --- Define Tokenizer Mappings ---
# Using class attributes for better organization within the Dataset
# TOKEN_TO_ID = {
#     '\\infty': 1, '\\ldots': 2, '\\times': 3, '\\theta': 4, '\\alpha': 5,
#     '\\gamma': 6, '\\lambda': 7, '\\sigma': 8, '\\cdot': 9, '\\frac{': 10,
#     '\\sqrt{': 11, '\\log_': 12, '\\neq': 13, '\\beta': 14, '\\phi': 15,
#     '\\div': 16, '\\geq': 17, '\\leq': 18, '\\sin': 19, '\\cos': 20,
#     '\\tan': 21, '\\mu': 22, '\\pi': 23, '\\pm': 24, '^{': 25,
#     '_{': 26, '(': 27, ')': 28, '{': 29, # Removed '}' : 30
#     '^': 31, '_': 32, '=': 33, '+': 34, '-': 35, '!': 36, '>': 37, '<': 38,
#     ' ': 39,  # Note: The tokenizer function skips spaces by default now
#     '0': 40, '1': 41, '2': 42, '3': 43, '4': 44, '5': 45, '6': 46, '7': 47,
#     '8': 48, '9': 49, 'A': 50, 'B': 51, 'C': 52, 'D': 53, 'E': 54, 'F': 55,
#     'G': 56, 'H': 57, 'L': 58, 'M': 59, 'N': 60, 'P': 61, 'R': 62, 'S': 63,
#     'T': 64, 'V': 65, 'X': 66, 'Y': 67, 'a': 68, 'b': 69, 'c': 70, 'd': 71,
#     'e': 72, 'f': 73, 'g': 74, 'h': 75, 'i': 76, 'j': 77, 'k': 78, 'l': 79,
#     'm': 80, 'n': 81, 'o': 82, 'p': 83, 'q': 84, 'r': 85, 's': 86, 't': 87,
#     'u': 88, 'v': 89, 'w': 90, 'x': 91, 'y': 92, 'z': 93,
# }
TOKEN_TO_ID = {
    '\\frac{'  : 0,
    '\\sqrt{'  : 1,
    '\\times'  : 2,
    '\\neq'    : 3,
    '\\geq'    : 4,
    '\\leq'    : 5,
    '\\pi'     : 6,
    '^{'       : 7,
    '_{'       : 8,
    '('        : 9,
    ')'        : 10,
    '^'        : 11,
    '_'        : 12,
    '='        : 13,
    '+'        : 14,
    '-'        : 15,
    '>'        : 16,
    '<'        : 17,
    '0'        : 18,
    '1'        : 19,
    '2'        : 20,
    '3'        : 21,
    '4'        : 22,
    '5'        : 23,
    '6'        : 24,
    '7'        : 25,
    '8'        : 26,
    '9'        : 27,
    'A'        : 28,
    'B'        : 29,
    'C'        : 30,
    'D'        : 31,
    'E'        : 32,
    'F'        : 33,
    'G'        : 34,
    'H'        : 35,
    'I'        : 36,
    'J'        : 37,
    'K'        : 38,
    'L'        : 39,
    'M'        : 40,
    'N'        : 41,
    'O'        : 42,
    'P'        : 43,
    'Q'        : 44,
    'R'        : 45,
    'S'        : 46,
    'T'        : 47,
    'U'        : 48,
    'V'        : 49,
    'W'        : 50,
    'X'        : 51,
    'Y'        : 52,
    'Z'        : 53,
    'a'        : 54,
    'b'        : 55,
    'c'        : 56,
    'd'        : 57,
    'e'        : 58,
    'f'        : 59,
    'g'        : 60,
    'h'        : 61,
    'i'        : 62,
    'j'        : 63,
    'k'        : 64,
    'l'        : 65,
    'm'        : 66,
    'n'        : 67,
    'o'        : 68,
    'p'        : 69,
    'q'        : 70,
    'r'        : 71,
    's'        : 72,
    't'        : 73,
    'u'        : 74,
    'v'        : 75,
    'w'        : 76,
    'x'        : 77,
    'y'        : 78,
    'z'        : 79,
    '<PAD>'    : 80,
    '<SOS>'    : 81,
    '<EOS>'    : 82,
    '<UNK>'    : 83
}

CONTEXTUAL_CLOSING_BRACES = {
    '\\frac{': 84,  # ID for closing brace of fraction numerator
    '{': 85,        # ID for closing brace of fraction denominator
    '\\sqrt{': 86,  # ID for closing brace of square root
    '_{': 87,       # ID for closing brace of subscript
    '^{': 88        # ID for closing brace of exponent
}


# Create reverse mappings
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}
ID_TO_CONTEXT = {v: k for k, v in CONTEXTUAL_CLOSING_BRACES.items()}

# Combine all IDs to find the range for vocab size
ALL_IDS = list(TOKEN_TO_ID.values()) + list(CONTEXTUAL_CLOSING_BRACES.values())
# Ensure PAD, SOS, EOS, UNK are included if defined
ALL_IDS.extend([TOKEN_TO_ID.get('<PAD>', 0), TOKEN_TO_ID.get('<SOS>', -1), TOKEN_TO_ID.get('<EOS>', -1), TOKEN_TO_ID.get('<UNK>', -1)])
ALL_IDS = [i for i in ALL_IDS if i >= 0] # Filter out potential -1 placeholders if not defined
VOCAB_SIZE = max(ALL_IDS) + 1 if ALL_IDS else 0 # +1 because IDs are typically 0-indexed or 1-indexed

# Sort tokens for efficient matching in tokenizer
# Exclude special tokens like <PAD> from sorting if they shouldn't be matched literally
SORTED_TOKENS = sorted([k for k in TOKEN_TO_ID.keys() if not k.startswith('<') and not k.endswith('>') ], key=len, reverse=True)

class MathHandwritingDataset(Dataset):
    """Handwriting dataset with LaTeX tokenization."""

    # --- Store tokenizer info as class attributes ---
    token_to_id = TOKEN_TO_ID
    contextual_closing_braces = CONTEXTUAL_CLOSING_BRACES
    id_to_token = ID_TO_TOKEN
    id_to_context = ID_TO_CONTEXT
    sorted_tokens = SORTED_TOKENS
    vocab_size = VOCAB_SIZE
    pad_token_id = TOKEN_TO_ID.get('<PAD>', 80)
    unk_token_id = TOKEN_TO_ID.get('<UNK>', 83) # Default UNK id
    # ---

    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        """
        Args:
            data_path (string): Path to the data folder (expecting strokes.npy and sentences.txt).
            split (string): 'train' or 'valid'.
            text_req (bool): If True, __getitem__ returns tokenized text sequences.
            debug (bool): If True, use a small subset of data.
            max_seq_len (int): Maximum sequence length for stroke data during data augmentation.
            data_aug (bool): If True, apply data augmentation (random cropping of strokes).
        """
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        # --- Load Stroke Data ---
        strokes = np.load(data_path + 'strokes.npy', allow_pickle=True, encoding='bytes')
        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths) if lengths else 0
        n_total = len(strokes)

        # Stroke Mask
        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        # Stroke Data Array
        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, seq_len in enumerate(lengths):
             if seq_len > 0: # Avoid index errors for empty strokes
                mask[i, :seq_len] = 1.
                data[i, :seq_len] = strokes[i]

        # --- Load and Tokenize Text Data ---
        try:
            with open(data_path + 'sentences.txt') as file:
                raw_texts = file.read().splitlines()
        except FileNotFoundError:
            print(f"Error: sentences.txt not found at {data_path}")
            raw_texts = [""] * n_total # Create empty texts if file missing, allows stroke-only use

        if len(raw_texts) != n_total:
            raise ValueError(f"Number of strokes ({n_total}) does not match number of sentences ({len(raw_texts)})")

        # Tokenize all texts using the char_to_idx method (which now handles tokens)
        tokenized_texts = [self.char_to_idx(text) for text in raw_texts]

        # Calculate token sequence lengths and max length
        token_lens = [len(tokens) for tokens in tokenized_texts]
        max_token_len = np.max(token_lens) if token_lens else 0

        # Token Mask
        token_mask_shape = (n_total, max_token_len)
        token_mask = np.zeros(token_mask_shape, dtype=np.float32)

        # Token ID Array (use a suitable integer type, e.g., int32)
        # Initialize with the padding token ID
        token_ids = np.full((n_total, max_token_len), self.pad_token_id, dtype=np.int32)

        # Populate token arrays
        for i, (tokens, t_len) in enumerate(zip(tokenized_texts, token_lens)):
            if t_len > 0: # Handle potentially empty token sequences
                token_mask[i, :t_len] = 1.
                token_ids[i, :t_len] = tokens

        # --- Shuffle Data ---
        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        token_ids = token_ids[idx_permute]
        token_mask = token_mask[idx_permute]
        # Keep raw texts if needed for debugging or inspection, shuffle them too
        # raw_texts_shuffled = [raw_texts[i] for i in idx_permute]

        # --- Debug Subset ---
        if debug:
            n_debug = min(64, n_total) # Ensure debug size isn't larger than total
            data = data[:n_debug]
            mask = mask[:n_debug]
            token_ids = token_ids[:n_debug]
            token_mask = token_mask[:n_debug]
            # raw_texts_shuffled = raw_texts_shuffled[:n_debug]

        # --- Train/Validation Split ---
        n_samples = data.shape[0]
        n_train = int(0.9 * n_samples)

        # Store stroke data temporarily for normalization calculation
        self._data = data # Keep reference to full data for potential normalization calculation

        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.token_ids = token_ids[:n_train]
            self.token_mask = token_mask[:n_train]
            # self.raw_texts = raw_texts_shuffled[:n_train] # Optional
            # Apply normalization (assuming these functions/Global exist)
            # if n_train > 0: # Avoid normalization on empty data
            #    Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(self.dataset)
            print(f"Train split: {self.dataset.shape[0]} samples")

        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.token_ids = token_ids[n_train:]
            self.token_mask = token_mask[n_train:]
            # self.raw_texts = raw_texts_shuffled[n_train:] # Optional
            # Apply normalization using training stats (assuming these functions/Global exist)
            # Need to ensure Global.train_mean/std are set (e.g., by instantiating train set first)
            # if n_samples > n_train and Global.train_mean is not None: # Avoid normalization on empty data or if mean/std not calculated
            #    self.dataset = valid_offset_normalization(Global.train_mean, Global.train_std, self.dataset)
            print(f"Validation split: {self.dataset.shape[0]} samples")
        else:
             raise ValueError(f"Invalid split name: {split}. Choose 'train' or 'valid'.")

        # Clean up temporary full dataset reference if no longer needed
        # If normalization happens outside, you might need to keep self._data
        # del self._data

    def char_to_idx(self, latex_expr):
        """
        Tokenizes a LaTeX expression string into a list of token IDs.
        Note: Despite the name, this method processes multi-character tokens.
        """
        tokens = []    # This will hold the final list of token IDs
        i = 0          # Current index in the input LaTeX string
        stack = []     # Stack to keep track of open tokens requiring contextual closing

        while i < len(latex_expr):
            # Skip spaces explicitly if needed (they have ID 39 but often ignored)
            if latex_expr[i] == ' ':
                 i += 1
                 continue # Or append token_to_id[' '] if spaces are significant

            match_found = False

            # Try to match the longest possible token from our sorted list
            for tok in self.sorted_tokens:
                # Check bounds before slicing
                if i + len(tok) <= len(latex_expr) and latex_expr[i:i+len(tok)] == tok:
                    token_id = self.token_to_id[tok]
                    tokens.append(token_id)
                    # If this token needs a contextual closing brace, push it onto the stack
                    if tok in self.contextual_closing_braces:
                         # Special case for \frac{ num { den } -> push '{' for denominator brace context
                         if tok == '\\frac{':
                              stack.append('{') # Expect denominator brace next
                         stack.append(tok) # Push the token itself for context
                    i += len(tok)
                    match_found = True
                    break # Stop searching once the longest match is found

            if match_found:
                continue # Proceed to the next character/token

            # Handle closing brace '}'
            if latex_expr[i] == '}':
                if stack:
                    opening = stack.pop() # Pop the last opening context
                    # Check if the popped item requires a specific closing brace ID
                    if opening in self.contextual_closing_braces:
                        tokens.append(self.contextual_closing_braces[opening])
                    else:
                        # This case might indicate mismatched braces or a simple '{' being closed.
                        # If a simple '{' has its own ID and doesn't require context,
                        # closing it should ideally not involve the context stack.
                        # Let's append UNK for unexpected '}' based on stack context.
                        tokens.append(self.unk_token_id)

                else:
                    # Unmatched closing brace '}' found
                    tokens.append(self.unk_token_id)
                i += 1
                continue

            # If no multi-character token matched, check single characters in the main dict
            char = latex_expr[i]
            if char in self.token_to_id:
                 token_id = self.token_to_id[char]
                 tokens.append(token_id)
                 # Check if this single char is an opening brace that needs contextual closing
                 if char in self.contextual_closing_braces:
                      stack.append(char)
            else:
                # Unknown character not in vocabulary
                tokens.append(self.unk_token_id)
            i += 1 # Move to next character

        # Check for unclosed items on stack at the end (malformed expression)
        if stack:
             # Append UNK for each unclosed item
             tokens.extend([self.unk_token_id] * len(stack))

        # # Filter out any -1 tokens if they represent errors and shouldn't be in the sequence
        # # This might be needed if unk_token_id was -1
        # tokens = [tok for tok in tokens if tok != -1]

        return np.array(tokens).astype(np.float32) # Return numpy array of token IDs

    def idx_to_char(self, token_ids):
        """
        Converts a sequence of token IDs back into a list of token strings.
        Note: Despite the name, this method returns token strings, not single characters.
        Args:
            token_ids: An iterable (list, numpy array, tensor) of token IDs.
        Returns:
            A list of strings, where each string is a token.
        """
        tokens = []
        for token_id in token_ids:
             # Ensure token_id is an integer (e.g., if input is tensor)
             if hasattr(token_id, 'item'): # Check if it's a tensor element
                 token_id = token_id.item()
             else:
                 token_id = int(token_id)

             if token_id == self.pad_token_id:
                 continue # Skip padding tokens in the output list

             if token_id in self.id_to_token:
                 tokens.append(self.id_to_token[token_id])
             elif token_id in self.id_to_context:
                 tokens.append('}') # All contextual closing IDs map back to '}'
             else:
                 # Handle unknown tokens
                 tokens.append('<UNK>') # Represent unknown IDs using the UNK string
        return np.array(tokens) # Return list of token strings

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        # --- Stroke Data Preparation ---
        stroke_data = self.dataset[idx]
        stroke_mask = torch.from_numpy(self.mask[idx])

        if self.data_aug:
            # --- Data Augmentation: Random Crop Strokes ---
            seq_len = int(stroke_mask.sum().item()) # Get actual length from mask
            start = 0
            end = self.max_seq_len

            if seq_len > self.max_seq_len:
                # If sequence is longer than max_seq_len, pick a random start
                start = np.random.randint(0, high=seq_len - self.max_seq_len + 1) # Inclusive high
                end = start + self.max_seq_len
                actual_cropped_len = self.max_seq_len
            elif seq_len > 0:
                 # If sequence is shorter or equal, use the actual length
                 end = seq_len
                 actual_cropped_len = seq_len
            else: # seq_len is 0
                 end = 0
                 actual_cropped_len = 0

            # Slice the data and mask, ensuring output tensors have size max_seq_len
            cropped_stroke = np.full((self.max_seq_len, 3), 0.0, dtype=np.float32) # Pad with 0.0
            cropped_mask = torch.zeros(self.max_seq_len, dtype=torch.float32)

            if actual_cropped_len > 0:
                cropped_stroke[:actual_cropped_len] = stroke_data[start:end, :]
                cropped_mask[:actual_cropped_len] = stroke_mask[start:end]

            stroke_data_tensor = torch.from_numpy(cropped_stroke)
            stroke_mask = cropped_mask # Use the cropped mask

            # Create input sequence (shifted target) - handle short/empty sequences
            input_seq = torch.zeros_like(stroke_data_tensor, dtype=torch.float32)
            if actual_cropped_len > 1:
                input_seq[1:actual_cropped_len, :] = stroke_data_tensor[:actual_cropped_len-1, :]

            target = stroke_data_tensor

            # Note: Text data is NOT augmented/cropped here, only strokes are.
            # If you need corresponding text cropping, that's more complex.
            if self.text_req:
                 token_ids = torch.from_numpy(self.token_ids[idx]).long()
                 token_mask = torch.from_numpy(self.token_mask[idx])
                 return (input_seq, target, stroke_mask, token_ids, token_mask)
            else:
                 return (input_seq, target, stroke_mask)

        else:
             # --- No Data Augmentation ---
            stroke_data_tensor = torch.from_numpy(stroke_data)
            input_seq = torch.zeros_like(stroke_data_tensor, dtype=torch.float32)
            # Shift data by one time step for input (predict next point from previous)
            seq_len = int(stroke_mask.sum().item())
            if seq_len > 1:
                input_seq[1:seq_len, :] = stroke_data_tensor[:seq_len-1, :]

            target = stroke_data_tensor

            if self.text_req:
                # --- Include Tokenized Text ---
                token_ids = torch.from_numpy(self.token_ids[idx]).long() # Use long for token IDs
                token_mask = torch.from_numpy(self.token_mask[idx])
                return (input_seq, target, stroke_mask, token_ids, token_mask)
            else:
                # --- Strokes Only ---
                return (input_seq, target, stroke_mask)
