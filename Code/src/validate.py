##################################################################################################
#               VALIDATE BEST ConvLSTM MODEL FOR PRECIPITATION NOWCASTING                        #
##################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from model import create_convlstm_model
from dataset import EfficientDataset
from utils import DBZ_THRESHOLD, NUM_TIMESTEPS, OUTPUT_DIR, OUTPUT_DIR_MODEL, TARGET_SIZE, denormalize_radar_data

def calculate_validation_metrics(y_true, y_pred):
    """
    Calculate validation metrics for both DBZ and VEL predictions
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Calculate DBZ metrics
    pred_dbz = denormalize_radar_data(y_pred[..., 0], 'DBZ')
    true_dbz = denormalize_radar_data(y_true[..., 0], 'DBZ')
    
    # Create mask for valid values
    valid_mask = (y_true[..., 0] != 0)  # 0 in normalized space means invalid
    
    # Apply threshold for precipitation
    pred_binary = (pred_dbz > DBZ_THRESHOLD) & valid_mask
    true_binary = (true_dbz > DBZ_THRESHOLD) & valid_mask
    
    # Calculate contingency table elements
    TP = np.sum((pred_binary == 1) & (true_binary == 1))
    FP = np.sum((pred_binary == 1) & (true_binary == 0))
    FN = np.sum((pred_binary == 0) & (true_binary == 1))
    
    # Calculate DBZ metrics
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0
    pod = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate VEL metrics
    pred_vel = denormalize_radar_data(y_pred[..., 1], 'VEL')
    true_vel = denormalize_radar_data(y_true[..., 1], 'VEL')
    valid_mask_vel = (y_true[..., 1] != 0)
    vel_mae = np.mean(np.abs(pred_vel[valid_mask_vel] - true_vel[valid_mask_vel]))
    
    return {
        'dbz_csi': csi,
        'dbz_far': far,
        'dbz_pod': pod,
        'vel_mae': vel_mae
    }

def main():
    # Load validation data using the generator
    val_data_file = os.path.join(OUTPUT_DIR, 'val_data.npz')
    val_generator = EfficientDataset(val_data_file, 1, NUM_TIMESTEPS, TARGET_SIZE)
    
    # Create model with same architecture
    input_shape = (NUM_TIMESTEPS, 81, 120, 120, 2)  # (time, height, lat, lon, features)
    model = create_convlstm_model(input_shape)
    
    # Load best model weights
    checkpoint_dir = os.path.join(OUTPUT_DIR_MODEL, 'checkpoints')
    # Find the most recent best model file
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_best.keras')]
    if not model_files:
        raise ValueError("No model checkpoint found!")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    model.load_weights(os.path.join(checkpoint_dir, latest_model))
    
    print(f"Loaded model: {latest_model}")
    
    # Evaluate model on full validation set
    print("Evaluating model on validation set...")
    metrics = model.evaluate(val_generator, verbose=1)
    print(f"Model Metrics:")
    for name, value in zip(model.metrics_names, metrics):
        print(f"{name}: {value:.4f}")
    
    # Generate predictions and calculate detailed metrics
    print("\nCalculating detailed metrics...")
    all_metrics = []
    for i in range(len(val_generator)):
        X, y_true = val_generator[i]
        y_pred = model.predict(X, verbose=0)
        batch_metrics = calculate_validation_metrics(y_true, y_pred)
        all_metrics.append(batch_metrics)
    
    # Average metrics across all batches
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\nDetailed Validation Metrics:")
    print(f"DBZ - CSI: {avg_metrics['dbz_csi']:.4f}")
    print(f"DBZ - FAR: {avg_metrics['dbz_far']:.4f}")
    print(f"DBZ - POD: {avg_metrics['dbz_pod']:.4f}")
    print(f"VEL - MAE: {avg_metrics['vel_mae']:.4f} m/s")
    
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
