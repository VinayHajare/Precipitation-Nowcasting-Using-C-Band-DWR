##################################################################################################
#               VALIDATE BEST ConvLSTM MODEL FOR PRECIPITATION NOWCASTING                        #
##################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from model import create_convlstm_model
from dataset import RadarDataset
from train import denormalize_radar_data, DBZ_THRESHOLD
from utils import output_dir, num_timesteps

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

def plot_prediction_examples(val_generator, model, num_examples=5):
    """
    Plot example predictions compared to ground truth
    
    Args:
        val_generator: Validation data generator
        model: Trained model
        num_examples: Number of examples to plot
    """
    # Get some validation samples
    for i in range(num_examples):
        X, y_true = val_generator[i]
        y_pred = model.predict(X)
        
        # Plot for the last timestep
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot DBZ
        true_dbz = denormalize_radar_data(y_true[0, -1, 40, :, :, 0], 'DBZ')  # Middle height level
        pred_dbz = denormalize_radar_data(y_pred[0, -1, 40, :, :, 0], 'DBZ')
        
        im1 = axes[0, 0].imshow(true_dbz, cmap='rainbow')
        axes[0, 0].set_title('True DBZ')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(pred_dbz, cmap='rainbow')
        axes[0, 1].set_title('Predicted DBZ')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot VEL
        true_vel = denormalize_radar_data(y_true[0, -1, 40, :, :, 1], 'VEL')
        pred_vel = denormalize_radar_data(y_pred[0, -1, 40, :, :, 1], 'VEL')
        
        im3 = axes[1, 0].imshow(true_vel, cmap='rainbow')
        axes[1, 0].set_title('True VEL')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(pred_vel, cmap='rainbow')
        axes[1, 1].set_title('Predicted VEL')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_example_{i}.png'))
        plt.close()

def main():
    # Load validation data using the generator
    val_data_file = os.path.join(output_dir, 'val_data.npz')
    val_generator = RadarDataset(val_data_file, batch_size=1, num_timesteps=num_timesteps)
    
    # Create model with same architecture
    input_shape = (num_timesteps, 81, 481, 481, 2)  # (time, height, lat, lon, features)
    model = create_convlstm_model(input_shape)
    
    # Load best model weights
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
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
    
    # Plot some example predictions
    print("\nGenerating prediction visualizations...")
    plot_prediction_examples(val_generator, model)
    
    print("\nValidation complete. Results and visualizations saved to output directory.")

if __name__ == "__main__":
    main()