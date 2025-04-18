# Anomaly_Detection using LSTM, GRU and Clustering methods

This repository provides various models (LSTM, GRU, and clustering techniques) for anomaly detection in sequential trace data. The goal is to detect abnormal behaviors by learning temporal patterns and evaluating the models using performance metrics such as Precision, Recall, and F1 Score.

---

## Features

- **Data Preprocessing**: Includes normalization, formatting for LSTM input, and train-validation split.
- **LSTM/GRU Models**: Deep learning models for time series anomaly detection with visualization.
- **Clustering Methods**: Additional techniques for unsupervised anomaly detection.
- **Evaluation Metrics**: Built-in metrics such as Precision, Recall, F1 Score, and Confusion Matrix.
- **Training & Validation Curves**: Loss curves are plotted to monitor training performance and detect overfitting.

---

## Folder Structure

```bash
Anomaly_Detection/
├── LSTM & GRU/                       # Deep learning models using LSTM and GRU
│   ├── libraries/                    # Contains reusable modules or helper scripts
│   ├── scalers/                      # Stores scalers used for data normalization
│   ├── trained_models/               # Pretrained model weights for reuse
│   ├── GRU_ID_MODEL.ipynb            # GRU model with only ID
│   ├── GRU_ID_TIMESTAMP_MODEL.ipynb  # GRU model using both ID and timestamp
│   ├── LSTM_ID_Model.ipynb           # LSTM model using only ID-based input
│   ├── LSTM_ID_TIMESTAMP_Model.ipynb # LSTM model using ID + timestamp input
│   ├── forecaster_id_test.ipynb      # Forecasting-based ID test
│   ├── gru_id_test.ipynb             # GRU evaluation on ID input
│   ├── gru_id_timestamp_test.ipynb   # GRU evaluation on ID + timestamp
│   ├── lstm_id_test.ipynb            # LSTM evaluation on ID input
│   └── lstm_id_timestamp_test.ipynb  # LSTM evaluation on ID + timestamp
├── clustering_methods/               # Includes clustering-based anomaly detection methods
├── dustminer/                        # Contains implemetation of dustminer
├── trained_minmax_scaler/            # Stores trained scalers used for normalizing input data
├── anomaly_detection.py              # Contains helper functions which are common for LSTM and GRU
├── utils.py                          # Contains the helper function like read_json, load_data
└── README.md                         # Detailed documentation
```


## Prerequisites

Make sure the following libraries are installed in your environment:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the required packages using below command:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---


## File Structure

- **LSTM_Model.ipynb**: Jupyter Notebook file containing the full implementation of the LSTM anomaly detection model.
- **lstm.py**: Contains the function which is used for merging detected sequences and comparing with ground truth data.
- **utils.py**: Contains helper functions for LSTM_Model.ipynb.

---

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/arjuncee123/Anomaly_Detection.git
   cd Anomaly_Detection
   ```

2. **Prepare Data**:
   - Place your zipped trace data file (`theft_protection.zip`) in the project directory.
   - Extract the zip file. After extracting, please ensure the folder structure and file formats match what the notebook expects.

3. **Run the Notebook**:
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook LSTM_Model.ipynb
     ```
   - Once the notebook is opened:
      1. Execute each cell in order to preprocess the data, train the model and evaluate results.
      2. Make sure all the dependencies are installed. The required libraries can be installed via the below command.
         ```bash
         pip install -r requirements.txt
         ```

4. **Train the Model**:
   - The notebook will handle the training process:
   - It uses an LSTM architecture suited for sequential time series data.
   - The notebook will automatically split the dataset into training and validation sets.
   - Training and validation loss plots will be displayed to monitor model performance over epochs.
   - We can modify the number of epochs and batch size to optimize the training based on these plots. We should stop the training if overfitting is observed.

5. **Evaluate the Model**:
   - Precision, Recall, F1 Score, and Confusion Matrix will be printed.
   - Visualizations of predictions and anomalies will be displayed.

---

## Output

- Training and validation loss curves.
- Anomaly detection results with identified anomalies in trace data.
- Evaluation metrics (Precision, Recall, F1 Score).
- Confusion matrix visualization.

---

## Supported Scenarios

- ID based anomaly detection
- Timestamp + ID forecasting and anomaly prediction

## Future Improvements

- Include Early Stopping to optimize training.
- Use additional anomaly detection thresholds dynamically.
- Add support for multi-threaded trace data.

---

## License

This project is licensed under the MIT License.

---
