import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import tensorflow as tf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import StadiaMapsTiles
from matplotlib.ticker import MultipleLocator, NullFormatter
from scipy.interpolate import griddata
from utils import (
    normalize_radar_data,
    denormalize_radar_data,
    downsample_data, 
    calculate_rainfall_intensity,
    DBZ_THRESHOLD,
    TARGET_SIZE,
    RADAR_PARAMS,
    NUM_TIMESTEPS,
    OUTPUT_DIR
)
from model import create_convlstm_model

class PrecipitationNowcaster:
    def __init__(self, model_path):
        """
        Initialize the nowcasting model.
        Args:
            model_path (type): Path to the best model checkpoint
        """
        # Input shape based on original preprocessing 
        input_shape = (NUM_TIMESTEPS, 81, 120, 120, 2) # (time, height, lat, lon, features)
        self.model = create_convlstm_model(input_shape)
        
        # Load the model weights
        self.model.load_weights(model_path)
    
    def preprocess_input_data(self, input_files):
        """
        Preprocess the input NetCDF files for the model prediction.
        
        Args:
            input_files (list): List of input NetCDF file paths for input timesteps
        
        Returns:
            numpy.ndarray: Preprocessed input data
        """
        input_data = []
        
        for file_path in input_files:
            ds = nc.Dataset(file_path)
            dbz = ds.variables['DBZH'][:].astype(np.float16)
            vel = ds.variables['VEL'][:].astype(np.float16)
            ds.close()
            
            # Normalize DBZ and VEL
            dbz_norm = normalize_radar_data(dbz, 'DBZ')
            vel_norm = normalize_radar_data(vel, 'VEL')
            
            # Stack normalized DBZ and VEL
            combined_data = np.stack([dbz_norm, vel_norm], axis=-1)
            input_data.append(np.squeeze(combined_data, axis=0))
        
        # Downsample and reshape the input data
        input_array = np.array(input_data)[np.newaxis, ...]
        input_downsampled = downsample_data(input_array, TARGET_SIZE)
        
        return input_downsampled
    
    def predict(self, input_files):
        """
        Predict next timestep precipitation and velocity
        
        Args:
            input_files (list): List of NetCDF file paths for input timesteps
        
        Returns:
            tuple: Predicted DBZ and velocity arrays
        """
        # Preprocess the input data
        input_data = self.preprocess_input_data(input_files)
        
        # Predict input data
        prediction = self.model.predict(input_data, verbose=0)
        
        # Denormalize the predicted data
        pred_dbz = denormalize_radar_data(prediction[0, :, :, :, :, 0], 'DBZ')
        pred_vel = denormalize_radar_data(prediction[0, :, :, :, :, 0], 'VEL')
        
        return pred_dbz, pred_vel
    
    def prepare_coordinates(self, orig_lat, orig_lon, orig_heights, target_size):
        """
        Prepare coordinates arrays for visualization, handling downsampling
        
        Args:
            orig_lat (numpy.ndarray): Original latitude values (481)
            orig_lon (numpy.ndarray): Original logitude values (481)
            orig_heights (numpy.ndarray): Original height values (81)
            target_size (numpy.ndarray): Target size after downsampling (120)
        
        Returns:
            tuple: Downsampled latitude, longitude, and original height arrays
        """
        # Create downsampled coordinate arrays
        lat_indices = np.linspace(0, len(orig_lat)-1, target_size, dtype=int)
        lon_indices = np.linspace(0, len(orig_lon)-1, target_size, dtype=int)
        
        downsampled_lat = orig_lat[lat_indices]
        downsampled_lon = orig_lon[lon_indices]
        
        return downsampled_lat, downsampled_lon, orig_heights
    
    def interpolate_to_original_grid(self, data, ds_lat, ds_lon, orig_lat, orig_lon):
        """
        Interpolate downsampled data back to original resolution for smoother visualization
        
        Args:
            data (numpy.ndarray): Downsampled data to interpolate
            ds_lat (numpy.ndarray): Downsampled latitude values
            ds_lon (numpy.ndarray): Downsampled longitude values
            orig_lat (numpy.ndarray): Original latitude values
            orig_lon (numpy.ndarray): Original longitude values
            
        Returns:
            numpy.ndarray: Interpolated data at original resolution
        """
        # Create coordinate meshgrids
        ds_lon_grid, ds_lat_grid = np.meshgrid(ds_lon, ds_lat)
        orig_lon_grid, orig_lat_grid = np.meshgrid(orig_lon, orig_lat)
        
        # Reshape coordinates for griddata
        points = np.column_stack((ds_lon_grid.flatten(), ds_lat_grid.flatten()))
        values = data.flatten()
        
        # Interpolate
        return griddata(
            points, values, 
            (orig_lon_grid, orig_lat_grid),
            method='cubic',
            fill_value=np.nan
        )
        
    def plot_enhanced_max_z(self, pred_dbz, orig_lat, orig_lon, orig_heights, save_path=None, rainfall_intensity=False):
        """
        Plot enhanced Max-Z visualization with proper resolution handling
        
        Args:
            pred_dbz (numpy.ndarray): Predicted DBZ values (1, 2, 81, 120, 120)
            orig_lat (numpy.ndarray): Original latitude values (481,)
            orig_lon (numpy.ndarray): Original longitude values (481,)
            orig_heights (numpy.ndarray): Height values (81,)
            save_path (str, optional): Path to save the plot
            rainfall_intensity (bool): Flag to plot rainfall intensity instead of DBZ
        """
                # Prepare coordinates
        ds_lat, ds_lon, heights = self.prepare_coordinates(
            orig_lat, orig_lon, orig_heights, TARGET_SIZE
        )
        
        # Process data
        ref1 = np.squeeze(pred_dbz, axis=0)
        if rainfall_intensity:
            ref1 = calculate_rainfall_intensity(ref1)
        
        # Calculate maximum projections
        ref2 = np.nanmax(ref1, axis=0)  # Max-Z plot
        refxz = np.nanmax(ref1, axis=2)  # Longitude-height cross section
        refyz = np.nanmax(ref1, axis=1)  # Latitude-height cross section
        
        # Interpolate Max-Z plot to original resolution for smoother visualization
        ref2_interp = self.interpolate_to_original_grid(ref2, ds_lat, ds_lon, orig_lat, orig_lon)
        
        # Create figure and axes
        plt.figure(figsize=(10, 10))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.2
        ax_xy = plt.axes((left, bottom, width, width), projection=ccrs.PlateCarree())
        ax_x = plt.axes((left, bottom + width, width, height))
        ax_y = plt.axes((left + width, bottom, height, width))
        ax_cb = plt.axes((left + width + height + 0.02, bottom, 0.02, width))

        # Configure formatters
        ax_x.xaxis.set_major_formatter(NullFormatter())
        ax_y.yaxis.set_major_formatter(NullFormatter())
        plt.sca(ax_xy)

        # Add geographical features
        ax_xy.add_feature(cfeature.LAND, edgecolor='black')
        ax_xy.add_feature(cfeature.COASTLINE)
        ax_xy.add_feature(cfeature.BORDERS, linestyle=':')
        ax_xy.add_feature(cfeature.LAKES, alpha=0.5)
        ax_xy.add_feature(cfeature.RIVERS)
        ax_xy.add_feature(cfeature.STATES, edgecolor='gray')

        # Add terrain
        stamen_terrain = StadiaMapsTiles(apikey='92c9d852-6357-413e-a8fd-9951a8f3e12d', style="stamen_terrain")
        ax_xy.add_image(stamen_terrain, 8)
        ax_xy.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        # Add radar circle
        radar_center = (np.mean(orig_lon), np.mean(orig_lat))
        radar_circle = plt.Circle(radar_center, radius=2, transform=ccrs.PlateCarree(),
                                color='red', fill=False, linewidth=2)
        ax_xy.add_artist(radar_circle)

        # Set plot parameters
        if rainfall_intensity:
            levels = range(0, 101, 10)
            cmap = 'YlGnBu'
            label = 'Rainfall Intensity (mm/hr)'
        else:
            levels = range(5, 55, 5)
            cmap = 'jet'
            label = 'Reflectivity (dBZ)'

        # Create plots
        xy = ax_xy.contourf(orig_lon, orig_lat, ref2_interp, 
                           levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
        cb = plt.colorbar(xy, cax=ax_cb, label=label)
        
        # Cross sections (using downsampled coordinates)
        ax_x.contourf(ds_lon, heights / 1000, refyz, levels=levels, cmap=cmap)
        ax_y.contourf(heights / 1000, ds_lat, refxz.T, levels=levels, cmap=cmap)

        # Labels and formatting
        ax_xy.set_xlabel('Longitude')
        ax_xy.set_ylabel('Latitude')
        ax_x.set_xlabel('')
        ax_x.set_ylabel('Elevation (km)')
        ax_y.set_ylabel('')
        ax_y.set_xlabel('Elevation (km)')

        ax_xy.xaxis.set_major_locator(MultipleLocator(1))
        ax_xy.yaxis.set_major_locator(MultipleLocator(1))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()

        
def main():
     # Configuration
     checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
     
     # Find the latest best model
     model_files =  [f for f in os.listdir(checkpoint_dir) if f.endswith('_best.keras')]
     if not model_files:
         raise ValueError("No model checkpoints found!")
     latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
     model_path = os.path.join(checkpoint_dir, latest_model)
     
     # Initialize the nowcaster
     nowcaster = PrecipitationNowcaster(model_path)
     
     # Input files
     input_files = [
         'Data/raw/1.nc',
         'Data/raw/2.nc',
     ]
     
     # Load original coordinate information
     with nc.Dataset(input_files[0]) as ds:
        orig_lat = ds.variables['latitude'][:]  # (481,)
        orig_lon = ds.variables['longitude'][:]  # (481,)
        orig_heights = ds.variables['height'][:]  # (81,)
     
     # Predict the next half-hour precipitation
     pred_dbz, pred_vel = nowcaster.predict(input_files)
     
     # Plot the MAX-Z
     nowcaster.plot_enhanced_max_z(
         pred_dbz,
         orig_lat,
         orig_lon,
         orig_heights, 
         save_path='max_z_plot.png'
     )
     
     # Plot the rainfall intensity
     nowcaster.plot_enhanced_max_z(
        pred_dbz, 
        orig_lat, 
        orig_lon, 
        orig_heights,
        save_path='rainfall_intensity_plot.png',
        rainfall_intensity=True
    )
     
if __name__ == '__main__':
    main()
            