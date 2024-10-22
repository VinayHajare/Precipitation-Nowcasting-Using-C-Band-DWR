##################################################################################################
#                           CREATES ConvLSTM FOR PRECIPITATION NOWCASTING                        #
##################################################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, ConvLSTM3D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
from utils import NUM_TIMESTEPS, TEMPORAL_RESOLUTION, RADAR_PARAMS


def create_convlstm_model(input_shape):
    """
    Define a ConvLSTM3D model for nowcasting using 3D volumetric radar data.
    Args:
        input_shape: Tuple of (time_steps, height, lat, lon, features)
    Returns:
        Compiled Keras model
    """
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))
    
    # First ConvLSTM3D block
    model.add(ConvLSTM3D(
        filters=64,
        kernel_size=(3, 3, 3),
        padding='same',
        return_sequences=True,
        activation='relu',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    
    # Second ConvLSTM3D block
    model.add(ConvLSTM3D(
        filters=64,
        kernel_size=(3, 3, 3),
        padding='same',
        return_sequences=True,
        activation='relu',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    
    # Third ConvLSTM3D block
    model.add(ConvLSTM3D(
        filters=32,
        kernel_size=(3, 3, 3),
        padding='same',
        return_sequences=True,
        activation='relu',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(TimeDistributed(Conv3D(
        filters=2,  # One channel each for DBZ and VEL
        kernel_size=(3, 3, 3),
        activation='sigmoid',  # Sigmoid because we normalized to [0,1]
        padding='same',
        kernel_initializer='he_normal'
    )))

    # Custom metric to handle invalid values
    def masked_mae(y_true, y_pred):
        """
        Calculate MAE only for valid values (non-zero in normalized space)
        """
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        mae = tf.abs(y_true - y_pred) * mask
        return tf.reduce_sum(mae) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
    
    def masked_mse(y_true, y_pred):
        """
        Calculate MSE only for valid values (non-zero in normalized space)
        """
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        mse = tf.square(y_true - y_pred) * mask
        return tf.reduce_sum(mse) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

    # Compile model with custom metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=masked_mse,
        metrics=[masked_mae]
    )
    
    return model