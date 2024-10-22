import numpy as np
import tensorflow as tf

class RadarDataset(tf.keras.utils.Sequence):
    def __init__(self, npz_file, batch_size, num_timesteps, **kwargs):
        super().__init__(**kwargs)
        
        # Load the npz file in memory-mapped mode (on disk)
        data = np.load(npz_file, mmap_mode='r')
        
        # Extract the data and labels
        self.data = data['data']
        self.labels = data['label']

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