##################################################################################################
#                   TRAIN ConvLSTM FOR PRECIPITATION NOWCASTING ON CUSTOM DATASET                #
##################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
from model import create_convlstm_model
from dataset import RadarDataset
from utils import OUTPUT_DIR, NUM_TIMESTEPS, RADAR_PARAMS
from datetime import datetime

# Model Hyperparameters
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.001
DBZ_THRESHOLD = 35.0  # dBZ threshold for precipitation

def denormalize_radar_data(normalized_data, variable_type):
    """Convert normalized values back to original radar units."""
    params = RADAR_PARAMS[variable_type]

    # Create a mask for invalid values (those that were set to 0 during normalization)
    valid_mask = normalized_data != 0

    # Initialize denormalized array with -999.0 for invalid values
    denormalized = np.full_like(normalized_data, -999.0, dtype=np.float32)

    # Denormalize only valid values
    denormalized[valid_mask] = (normalized_data[valid_mask] * (params['max'] - params['min'])) + params['min']

    # Update the values where the mask is False to -999.0
    denormalized.data[denormalized.data == 0.0] = -999.0

    return denormalized

class ValidationMetrics(Callback):
    """Custom callback for calculating radar-specific validation metrics"""
    
    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions and true values
        val_predictions = self.model.predict(self.val_generator)
        val_true = np.concatenate([y for _, y in self.val_generator], axis=0)
        
        # Calculate metrics for DBZ
        dbz_metrics = self.calculate_dbz_metrics(
            val_predictions[..., 0],  # DBZ channel
            val_true[..., 0]         # DBZ channel
        )
        
        # Calculate metrics for VEL
        vel_metrics = self.calculate_vel_metrics(
            val_predictions[..., 1],  # VEL channel
            val_true[..., 1]         # VEL channel
        )
        
        # Log all metrics
        logs.update(dbz_metrics)
        logs.update(vel_metrics)
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}")
        print(f"DBZ - CSI: {dbz_metrics['dbz_csi']:.4f}, "
              f"FAR: {dbz_metrics['dbz_far']:.4f}, "
              f"POD: {dbz_metrics['dbz_pod']:.4f}")
        print(f"VEL - MAE: {vel_metrics['vel_mae']:.4f} m/s")
    
    def calculate_dbz_metrics(self, pred, true):
        """Calculate DBZ-specific metrics (CSI, FAR, POD)"""
        # Denormalize
        pred_dbz = denormalize_radar_data(pred, 'DBZ')
        true_dbz = denormalize_radar_data(true, 'DBZ')
        
        # Create mask for valid values
        valid_mask = (true != 0)  # 0 in normalized space means invalid
        
        # Apply threshold for precipitation
        pred_binary = (pred_dbz > DBZ_THRESHOLD) & valid_mask
        true_binary = (true_dbz > DBZ_THRESHOLD) & valid_mask
        
        # Calculate contingency table elements
        TP = np.sum((pred_binary == 1) & (true_binary == 1))
        FP = np.sum((pred_binary == 1) & (true_binary == 0))
        FN = np.sum((pred_binary == 0) & (true_binary == 1))
        
        # Calculate metrics
        csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        far = FP / (TP + FP) if (TP + FP) > 0 else 0
        pod = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        return {
            'dbz_csi': csi,
            'dbz_far': far,
            'dbz_pod': pod
        }
    
    def calculate_vel_metrics(self, pred, true):
        """Calculate velocity-specific metrics"""
        # Denormalize
        pred_vel = denormalize_radar_data(pred, 'VEL')
        true_vel = denormalize_radar_data(true, 'VEL')
        
        # Create mask for valid values
        valid_mask = (true != 0)  # 0 in normalized space means invalid
        
        # Calculate MAE for valid values only
        mae = np.mean(np.abs(pred_vel[valid_mask] - true_vel[valid_mask]))
        
        return {'vel_mae': mae}

def setup_callbacks(val_generator):
    """Setup training callbacks"""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint to save best model
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'model_{timestamp}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard for monitoring training
        TensorBoard(
            log_dir=os.path.join(log_dir, timestamp),
            histogram_freq=1,
            update_freq='epoch'
        ),
        
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Custom validation metrics
        ValidationMetrics(val_generator)
    ]
    
    return callbacks

def plot_training_history(history):
    """Plot training history"""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='train')
    axes[0, 0].plot(history.history['val_loss'], label='validation')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Plot MAE
    axes[0, 1].plot(history.history['masked_mae'], label='train')
    axes[0, 1].plot(history.history['val_masked_mae'], label='validation')
    axes[0, 1].set_title('Masked MAE')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Plot DBZ metrics
    axes[1, 0].plot(history.history['dbz_csi'], label='CSI')
    axes[1, 0].plot(history.history['dbz_far'], label='FAR')
    axes[1, 0].plot(history.history['dbz_pod'], label='POD')
    axes[1, 0].set_title('DBZ Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Plot VEL metrics
    axes[1, 1].plot(history.history['vel_mae'], label='MAE')
    axes[1, 1].set_title('Velocity MAE')
    axes[1, 1].set_ylabel('MAE (m/s)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

def main():
    # Load data
    train_data_file = os.path.join(OUTPUT_DIR, 'train_data.npz')
    val_data_file = os.path.join(OUTPUT_DIR, 'val_data.npz')

    # Create data generators
    train_generator = RadarDataset(train_data_file, BATCH_SIZE, NUM_TIMESTEPS)
    val_generator = RadarDataset(val_data_file, BATCH_SIZE, NUM_TIMESTEPS)

    # Create model
    input_shape = (NUM_TIMESTEPS, 81, 481, 481, 2)  # (time, height, lat, lon, features)
    model = create_convlstm_model(input_shape)
    model.summary()

    # Setup callbacks
    callbacks = setup_callbacks(val_generator)

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()