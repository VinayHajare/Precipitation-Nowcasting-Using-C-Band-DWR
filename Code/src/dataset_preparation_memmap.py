from cProfile import label
import os
import numpy as np
import netCDF4 as nc
from glob import glob
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from tqdm import tqdm
import gc
from utils import DATA_DIR, OUTPUT_DIR, NUM_TIMESTEPS, extract_timestamp, load_and_preprocess_data, normalize_radar_data, save_npz

def create_sequences_from_files(file_list, num_timesteps, max_time_gap=timedelta(minutes=20)):
    """Create sequences from files with radar-specific processing."""
    sequence_shape = (num_timesteps, 81, 481, 481, 2)
    num_sequences = len(file_list) // (num_timesteps * 2)
    
    sequences_path = os.path.join(OUTPUT_DIR, 'sequences.dat')
    labels_path = os.path.join(OUTPUT_DIR, 'labels.dat')
    
    # Using float32 for better precision
    sequences_memmap = np.memmap(sequences_path, dtype=np.float32, mode='w+', 
                                shape=(num_sequences,) + sequence_shape)
    labels_memmap = np.memmap(labels_path, dtype=np.float32, mode='w+', 
                             shape=(num_sequences,) + sequence_shape)
    
    sequence_count = 0
    i = 0
    
    with tqdm(total=len(file_list), desc="Processing Files") as pbar:
        while i < len(file_list) - num_timesteps * 2:
            # Collect files for sequence
            x_sequence_files = []
            y_label_files = []
            timestamps = []
            
            # Build sequence
            for j in range(i, len(file_list)):
                current_timestamp = extract_timestamp(file_list[j])
                if not timestamps:
                    x_sequence_files.append(file_list[j])
                    timestamps.append(current_timestamp)
                else:
                    time_diff = current_timestamp - timestamps[-1]
                    if time_diff <= max_time_gap:
                        if len(x_sequence_files) < num_timesteps:
                            x_sequence_files.append(file_list[j])
                            timestamps.append(current_timestamp)
                        elif len(y_label_files) < num_timesteps:
                            y_label_files.append(file_list[j])
                            timestamps.append(current_timestamp)
                    else:
                        break
                
                if len(x_sequence_files) == num_timesteps and len(y_label_files) == num_timesteps:
                    break
            
            # Process sequence if complete
            if len(x_sequence_files) == num_timesteps and len(y_label_files) == num_timesteps:
                sequence_data = []
                
                # Process input sequence
                for file_path in x_sequence_files:
                    dbz, vel = load_and_preprocess_data(file_path)
                    if dbz is None or vel is None:
                        sequence_data = None
                        break
                    
                    # Normalize each variable
                    dbz_norm = normalize_radar_data(dbz, 'DBZ')
                    vel_norm = normalize_radar_data(vel, 'VEL')
                    
                    # Stack normalized data
                    combined_data = np.stack((dbz_norm, vel_norm), axis=-1)
                    sequence_data.append(np.squeeze(combined_data, axis=0))
                
                # If sequence processing successful, process labels
                if sequence_data is not None and len(sequence_data) == num_timesteps:
                    label_sequence_data = []
                    
                    for file_path in y_label_files:
                        dbz_label, vel_label = load_and_preprocess_data(file_path)
                        if dbz_label is None or vel_label is None:
                            label_sequence_data = None
                            break
                        
                        dbz_label_norm = normalize_radar_data(dbz_label, 'DBZ')
                        vel_label_norm = normalize_radar_data(vel_label, 'VEL')
                        
                        combined_label = np.stack((dbz_label_norm, vel_label_norm), axis=-1)
                        label_sequence_data.append(np.squeeze(combined_label, axis=0))
                    
                    # Save sequence if both input and label processing successful
                    if label_sequence_data is not None and len(label_sequence_data) == num_timesteps:
                        sequences_memmap[sequence_count] = np.array(sequence_data, dtype=np.float32)
                        labels_memmap[sequence_count] = np.array(label_sequence_data, dtype=np.float32)
                        sequence_count += 1
            
            i += 1
            pbar.update(1)
            gc.collect()
    
    return sequences_memmap[:sequence_count], labels_memmap[:sequence_count]

# Main execution
if __name__ == "__main__":
    # Collect all NetCDF files
    files = glob(os.path.join(DATA_DIR, '**', '*.nc'), recursive=True)
    files = sorted(files, key=lambda x: extract_timestamp(x))
    print(f"Found {len(files)} NetCDF files.")
    
    # Process files to create sequences
    all_data, all_labels = create_sequences_from_files(files, NUM_TIMESTEPS)
    
    if all_data is None or all_labels is None:
        print("No data was processed successfully. Please check your input files and error messages above.")
    else:
        print(f"Processed data shape: {all_data.shape}")
        print(f"Processed labels shape: {all_labels.shape}")
        
        # Split data into train and validation sets
        indices = np.arange(all_data.shape[0])
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Save training data
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]
        save_npz(train_data, train_labels, os.path.join(OUTPUT_DIR, 'train_data.npz'))
        
        # Save validation data
        val_data = all_data[val_indices]
        val_labels = all_labels[val_indices]
        save_npz(val_data, val_labels, os.path.join(OUTPUT_DIR, 'val_data.npz'))
        
        print(f"Dataset preparation complete. Training samples: {train_data.shape[0]}, Validation samples: {val_data.shape[0]}")