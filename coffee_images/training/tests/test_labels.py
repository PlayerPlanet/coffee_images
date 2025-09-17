import os
import numpy as np
from coffee_images.training.coffee import CoffeeDataset

def test_label_types():
    # Adjust path as needed
    data_dir = "coffee_images/training/labels"
    dataset = CoffeeDataset(data_dir)
    for i in range(len(dataset)):
        _, label = dataset[i]
        # Convert to numpy if tensor
        if hasattr(label, 'numpy'):
            label = label.numpy()
        assert np.issubdtype(label.dtype, np.integer), f"Label dtype is not integer: {label.dtype}"
        unique_vals = np.unique(label)
        assert set(unique_vals).issubset({0, 1}), f"Label contains values other than 0 and 1: {unique_vals}"