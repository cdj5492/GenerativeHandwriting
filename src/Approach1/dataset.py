import copy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    """
    A Dataset wrapper for n-dimensional numpy arrays.
    All arrays must have the same leading dimension.
    
    Args:
        columns: List of names corresponding to each data array.
        data: List of numpy arrays corresponding to the columns.
        indices: (optional) An array of indices to select a subset of the data.
    """
    def __init__(self, columns, data, indices=None):
        assert len(columns) == len(data), "Number of columns must match number of data arrays."
        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, "All data arrays must have the same first dimension."
        
        self.columns = columns
        self.data = data
        self.length = lengths[0]
        self.indices = indices if indices is not None else np.arange(self.length)
        
        # Create a dict for convenience
        self.data_dict = dict(zip(self.columns, self.data))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Return a dict with a copy of each element for this sample
        sample = {col: self.data_dict[col][real_idx].copy() for col in self.columns}
        return sample

    def train_test_split(self, train_size, random_state=None, stratify=None):
        """
        Split the dataset into train and test subsets.
        """
        train_idx, test_idx = train_test_split(
            self.indices,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify
        )
        train_dataset = NumpyDataset(self.columns, self.data, indices=train_idx)
        test_dataset = NumpyDataset(self.columns, self.data, indices=test_idx)
        return train_dataset, test_dataset
