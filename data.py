import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification

def load_hospital_data(hospital_id, num_samples=5000):

    # Non-IID shift per hospital
    X, y = make_classification(
        n_samples=num_samples,
        n_features=20,
        n_informative=15,
        n_classes=2,
        flip_y=0.03,
        class_sep=1.0 + hospital_id * 0.2
    )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader