import numpy as np
from tensorflow.keras.callbacks import Callback
from utils import denormalize_radar_data, DBZ_THRESHOLD

 
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