##################################################################################################
#                   TRAIN ConvLSTM FOR PRECIPITATION NOWCASTING ON CUSTOM DATASET                #
##################################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from model import create_convlstm_model
from datetime import datetime
from dataset import EfficientDataset
from utils import OUTPUT_DIR, OUTPUT_DIR_LOGS, OUTPUT_DIR_MODEL, TARGET_SIZE
from custom_validation_callback import ValidationMetrics

# Model Hyperparameters
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.001

def setup_callbacks(val_generator):
    """Setup training callbacks"""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(OUTPUT_DIR_MODEL, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(OUTPUT_DIR_LOGS, 'logs')
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

    # Create datasets
    train_dataset = EfficientDataset(train_data_file, BATCH_SIZE, 2, TARGET_SIZE)
    val_dataset = EfficientDataset(val_data_file, BATCH_SIZE, 2, TARGET_SIZE)

    # Get downsampled shape
    X_sample, _ = train_dataset[0]
    input_shape = X_sample.shape[1:]
    print(f"Downsampled input shape: {input_shape}")

    # Create model
    #original_shape = (NUM_TIMESTEPS, 81, 481, 481, 2)  # (time, height, lat, lon, features)
    model = create_convlstm_model(input_shape)
    model.summary()

    # Setup callbacks
    callbacks = setup_callbacks(val_dataset)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()