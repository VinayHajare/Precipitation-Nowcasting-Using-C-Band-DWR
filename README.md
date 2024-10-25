---

# 🌧️ Precipitation Nowcasting with C-Band Doppler Radar Data 🌦️

## 🚀 Project Overview

This project utilizes deep learning to **nowcast precipitation** over the next half-hour using **C-Band Doppler radar data** from the TERLS radar system. By leveraging ConvLSTM3D models, it processes volumetric radar scans to predict short-term precipitation and wind intensity. The model provides **timely and localized precipitation insights**, which are essential for meteorology, disaster management, and early warning systems.

---

### 📂 Project Structure

- **Data**: Contains raw and processed radar data in `.nc` and `.npz` formats.
- **Code**:
  - `dataset_preparation_memmap.py`: Memory-mapped dataset preparation for handling large datasets.
  - `model.py`: Defines a `ConvLSTM3D` model with custom loss and metrics functions.
  - `train.py`: Training script with callbacks and metrics monitoring.
  - `validate.py`: Model evaluation script using radar-specific metrics.
  - `utils.py`: Utility functions for data normalization, denormalization, and metrics.
  - `dataset.py`: A custom dataset class to handle huge high-dimensional Radar data.
  - `custom_validation_callback.py`: A custom training callback to evalute model on Radar specific metrics.
- **Notebooks**: For exploratory data analysis, visualization, and data inspections.
- **README.md**: Project documentation.

---

### 📌 Key Features

- **Memory-Mapped Data Handling** 🧠: Efficient data loading and preprocessing using memory-mapped files to accommodate large radar datasets.
- **ConvLSTM3D Model** 🏗️: Leverages the spatial and temporal dependencies in radar data to predict precipitation.
- **Custom Metrics** 📊: Includes meteorological metrics like **CSI**, **FAR**, and **POD** for model performance evaluation.
- **Mixed Precision Training** 💪: Reduces memory usage while preserving model accuracy.
- **Progressive Data Loading** 🐢: Batch-wise data streaming & processing with `tf.keras.utils.Sequence` to optimize memory and computational efficiency.

---

### 🔧 Installation and Setup

#### Prerequisites

1. **Python 3.7+**
2. **TensorFlow 2.x**
3. **NetCDF4 & Xarray** for radar data handling
4. **Other dependencies** (see `requirements.txt`)

#### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VinayHajare/Precipitation-Nowcasting-Using-C-Band-DWR.git
   cd Precipitation-Nowcasting-Using-C-Band-DWR
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   - Place raw `.nc` files in the `Data/raw` directory.
   - Run the dataset preparation script:
     ```bash
     bash Code/scripts/prepare.sh
     ```
   - This will create memory-mapped `.dat` files and save processed `.npz` files in the `Data/dataset` directory.

4. **Model Training**:
   ```bash
   bash Code/scripts/train.sh
   ```

5. **Model Evaluation**:
   ```bash
   bash Code/scripts/validate.sh
   ```

---

### 🏗️ Model Architecture

The model is based on **ConvLSTM3D** layers, designed to capture spatial and temporal patterns in radar scans.

```plaintext
[Input] → [ConvLSTM3D Layers] → [TimeDistributed Conv3D] → [Output]
```

- **Input**: Volumetric radar data with reflectivity (`DBZ`) and radial velocity (`VEL`) channels.
- **Output**: Predicted radar fields for the next half-hour, along with rainfall intensity.

---

### 📊 Custom Metrics

This project incorporates radar-specific metrics for assessing model performance:

- **CSI (Critical Success Index)**: Measures the accuracy of precipitation detection.
- **FAR (False Alarm Rate)**: Indicates the proportion of false positives.
- **POD (Probability of Detection)**: Represents the model's sensitivity.
- **MAE for VEL**: Measures the model's accuracy in predicting radial velocity.

### 📈 Training and Evaluation

- **Mixed Precision Training**: Enabled for efficient memory usage without sacrificing model accuracy.
- **Batch Size**: Optimized for high-dimensional radar data.
- **Callbacks**:
  - **ModelCheckpoint**: Saves the best model based on validation metrics.
  - **EarlyStopping**: Stops training if validation performance plateaus.
  - **TensorBoard**: Logs training and validation performance for visualization.

---

### 🧰 Utilities and Helper Functions

- **Data Normalization**: Scales `DBZ` and `VEL` values for efficient training.
- **Data Denormalization**: Converts normalized predictions back to original units for interpretation.
- **Metrics Calculation**: Custom functions for CSI, FAR, POD, and MAE.
- **Extract Timestamps**: Extract timestamps from NetCDF files for temporal-continuty.
- **Load & Preprocess Data**: Load and preprocess Radar data.
- **Downsample Data**: Downsample Radar data at the time of training.
- **Calculate Rainfall Intesity**: Calculate rainfall intensity based on radar data.

---

### 📌 Example Usage

Here's an example of loading, processing, and training the model with the provided code:

```python
# Load and preprocess data
!python dataset_preparation_memmap.py

# Train model
!python train.py

# Evaluate model
!python validate.py
```

---

### 📋 Future Improvements

- **Enhanced Model Tuning**: Hyperparameter optimization for improved accuracy.
- **Extended Evaluation Metrics**: Additional radar-specific metrics for better evaluation.
- **Additional Data Sources**: Integrate multiple radar sources for robust predictions.

---

### 📝 Acknowledgments

This project was developed as part of the ISRO Hackathon. The radar data is sourced from the **Thumba Equatorial Rocket Launching Station (TERLS)**, with support from **Space Applications Centre, ISRO**.

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🤝 Contributing

Feel free to fork this project, open issues, or submit pull requests. Contributions are welcome to help improve nowcasting capabilities and enhance model performance! ✨

---

### 🌐 Connect

For more projects like this, follow [Vinay Hajare](https://vinayhajare.engineer).

Happy nowcasting! 🌦️
