##################################################################################################
#                            DEFINES THE UTILITY METHODS & Constants                             #
##################################################################################################

import numpy as np
from datetime import datetime
import netCDF4 as nc
from scipy.ndimage import zoom

# Define directories
DATA_DIR = 'Data/raw'
OUTPUT_DIR = 'Data/dataset'
OUTPUT_DIR_MODEL = 'Data/models'
OUTPUT_DIR_LOGS = 'Data/logs'

TARGET_SIZE = (120, 120)  # Reduced spatial dimensions
DBZ_THRESHOLD = 20.0  # dBZ threshold for precipitation

# Define temporal resolution (10 minutes) and number of timesteps (6 for 1-hour nowcasting)
TEMPORAL_RESOLUTION = 15
NUM_TIMESTEPS = 2

# Define radar parameters
RADAR_PARAMS = {
    'DBZ': {
        'min': -30.0,  # Minimum detectable DBZ
        'max': 90.0,   # Maximum DBZ for severe precipitation
        'invalid': -999.0  # Invalid value indicator
    },
    'VEL': {
        'min': -110.0,  # Maximum negative velocity (m/s)
        'max': 80.0,   # Maximum positive velocity (m/s)
        'invalid': -999.0  # Invalid value indicator
    }
}

def save_npz(data_list, label_list, filename):
    """Save processed data to npz file."""
    np.savez_compressed(filename, data=data_list, label=label_list)


def normalize_radar_data(data, variable_type):
    """
    Normalize radar data based on variable type (DBZ or VEL).

    Parameters:
        data: Input radar data array
        variable_type: 'DBZ' or 'VEL'

    Returns:
        normalized: Array with values between 0 and 1, invalid values as 0
    """
    params = RADAR_PARAMS[variable_type]

    # Create a mask for invalid values (those equal to the invalid value)
    invalid_mask = np.isclose(data, params['invalid'])

    # Create a new array for normalized data
    normalized = np.zeros_like(data, dtype=np.float16)

    # Set invalid values to 0
    normalized[invalid_mask] = 0

    # Handle valid values (everything that's not -999)
    valid_mask = ~invalid_mask

    if np.any(valid_mask):
        valid_data = data[valid_mask]

        # Shift the minimum value in the valid data to 0
        valid_data_shifted = valid_data - params['min']

        # Clip the shifted valid data to the [0, max] range
        valid_data_clipped = np.clip(valid_data_shifted, 0, params['max'] - params['min'])

        # Normalize the clipped valid data to the [0, 1] range
        normalized[valid_mask] = valid_data_clipped / (params['max'] - params['min'])

        # Validation check: All normalized values should be in the range [0, 1]
        assert np.all((normalized[valid_mask] >= 0) & (normalized[valid_mask] <= 1)), \
            f"Normalization failed: values outside [0,1] range for {variable_type}"

        # Print some statistics for debugging
        #print(f"\n{variable_type} Statistics:")
        #print(f"Valid values range: [{np.min(normalized[valid_mask]):.3f}, {np.max(normalized[valid_mask]):.3f}]")
        #print(f"Percentage of valid values: {np.mean(valid_mask) * 100:.2f}%")

    # Return the normalized array (with invalid values staying as 0)
    return normalized


def denormalize_radar_data(normalized_data, variable_type):
    """Convert normalized values back to original radar units.
    
    Args:
        normalized_data: numpy.ndarray or numpy.ma.MaskedArray
            The normalized radar data to be denormalized
        variable_type: str
            The type of radar variable (e.g., 'DBZ', 'VEL')
            
    Returns:
        numpy.ndarray
            Denormalized radar data with -999.0 for invalid values
    """
    params = RADAR_PARAMS[variable_type]
    
    # Convert MaskedArray to regular array if needed
    if isinstance(normalized_data, np.ma.MaskedArray):
        valid_mask = ~normalized_data.mask
        data_array = normalized_data.data
    else:
        # For regular ndarray, use 0 as the invalid value marker
        valid_mask = normalized_data != 0
        data_array = normalized_data
    
    # Initialize output array with fill value
    denormalized = np.full_like(data_array, -999.0, dtype=np.float16)
    
    # Denormalize only valid values
    denormalized[valid_mask] = (data_array[valid_mask] * (params['max'] - params['min'])) + params['min']
    
    return denormalized


def load_and_preprocess_data(file_path):
    """Load NetCDF file and preprocess it into a usable format."""
    try:
        ds = nc.Dataset(file_path)
        dbz = ds.variables['DBZ'][:].astype(np.float16)
        vel = ds.variables['VEL'][:].astype(np.float16)
        ds.close()
        return dbz, vel
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None


def extract_timestamp(filename):
    """Extract timestamp from the filename."""
    try:
        date_str = filename.split('_')[1]
        time_str = filename.split('_')[2]
        datetime_str = date_str + time_str
        return datetime.strptime(datetime_str, '%d%b%Y%H%M%S')
    except Exception as e:
        print(f"Error extracting timestamp from {filename}: {str(e)}")
        return None


def calculate_rainfall_intensity(dbz_values):
    """
    Calculate rainfall intensity from DBZ values using the Marshall-Palmer Z-R relationship.
    
    Args:
        dbz_values: Array of DBZ (radar reflectivity) values
        
    Returns:
        Array of rainfall intensity values in mm/hr
        
    Notes:
        Uses Z = aR^b relationship where:
        Z = 10^(DBZ/10) [mm^6/m^3]
        R = rainfall rate [mm/hr]
        a = 267.0 (empirical coefficient)
        b = 1.3 (empirical coefficient)
        
        Therefore: R = (Z/a)^(1/b)
    """
    # Marshall-Palmer coefficients
    a = 267.0  # Standard coefficient for TERLS
    b = 1.3  # Standard exponent for TERLS
    
    # Convert DBZ to Z (mm^6/m^3)
    Z = np.power(10, dbz_values / 10)
    
    # Calculate rainfall rate (mm/hr)
    rainfall_intensity = np.power(Z / a, 1 / b)
    
    # Set invalid or negative values to 0
    rainfall_intensity = np.where(dbz_values > 0, rainfall_intensity, 0)
    
    return rainfall_intensity


def downsample_data(data, target_size=(120, 120)):
    """
    Downsample spatial dimensions using scipy's zoom

    Args:
        data: Array of shape (samples, timesteps, height, lat, lon, features)
        target_size: Tuple of (new_lat, new_lon)
    """
    samples, timesteps, height, lat, lon, features = data.shape

    # Calculate zoom factors
    lat_factor = target_size[0] / lat
    lon_factor = target_size[1] / lon

    # Initialize output array
    output = np.zeros((samples, timesteps, height, target_size[0], target_size[1], features))

    # Process each sample and feature
    for i in range(samples):
        for t in range(timesteps):
            for f in range(features):
                # Extract 3D volume
                volume = data[i, t, :, :, :, f]

                # Downsample using scipy zoom
                downsampled = zoom(volume, (1, lat_factor, lon_factor), order=1)

                # Store result
                output[i, t, :, :, :, f] = downsampled

    return output.astype(np.float32)