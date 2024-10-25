import numpy as np
import tensorflow as tf
from utils import downsample_data

class RadarDataset(tf.keras.utils.Sequence):
    def __init__(self, npz_file, batch_size, num_timesteps, **kwargs):
        super().__init__(**kwargs)
        
        # Load the npz file in memory-mapped mode (on disk)
        data = np.load(npz_file, mmap_mode='r')
        
        # Extract the data and labels
        self.data = data['data'].astype(np.float16)
        self.labels = data['label'].astype(np.float16)

        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.indices = np.arange(self.data.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        # Use memory-mapped data access to avoid loading the entire dataset into memory
        X = self.data[batch_indices]
        y = self.labels[batch_indices]

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    

class EfficientDataset(tf.keras.utils.Sequence):
    def __init__(self, npz_file, batch_size, num_timesteps, target_size=(120, 120), **kwargs):
        super().__init__(**kwargs)
        # Load memory-mapped data
        self.data_mmap = np.load(npz_file, mmap_mode='r')

        # Get data shapes
        self.data = self.data_mmap['data']
        self.labels = self.data_mmap['label']

        print(f"Original data shape: {self.data.shape}")

        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.data))

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load batch data
        X = self.data[batch_indices].astype(np.float32)
        y = self.labels[batch_indices].astype(np.float32)

        # Downsample batch
        X = downsample_data(X, self.target_size)
        y = downsample_data(y, self.target_size)

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
