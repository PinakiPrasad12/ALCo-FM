# ALCo-FM: Adaptive Long-Context Foundation Model for Accident Prediction

A research paper and accompanying codebase presenting **ALCo-FM**—the first foundation-model–based framework that (i) adaptively selects temporal context based on uncertainty, (ii) fuses numerical, spatial, and visual inputs, and (iii) produces calibrated, interpretable accident risk forecasts.

---

## Overview

Adaptive Long-Context Foundation Model (ALCo-FM) introduces:
- **Adaptive Temporal Context Selection**: Dynamically chooses history window \(w\in\{1,3,6\}\) hours based on a pre-score uncertainty \(u\) from a single-pass foundation embedding.
- **Uncertainty-Aware Fusion**: Uses Monte Carlo dropout for calibrated risk estimates and to weight numerical, visual, and spatial modalities.
- **Pretraining & Minimal-Data Fine-Tuning**: Pretrained on 15 U.S. cities, then fine-tuned with minimal data on 3 unseen cities, achieving robust generalization.
- **State-of-the-Art Performance**: Yields 0.93 accuracy and 0.91 F1, outperforming 20+ strong baselines in large-scale urban risk forecasts.
- **Interpretable Risk Maps**: Outputs spatial risk assessments highlighting low-risk (green) and high-risk (red) areas.

---

## Key Contributions

1. **Adaptive Long-Context Mechanism**  
   Uncertainty-driven selection of temporal history for efficient forecasting.

2. **Foundation-Model Fusion**  
   Leverages a pre-trained transformer to embed and fuse heterogeneous inputs.

3. **Calibration via Monte Carlo Dropout**  
   Provides well-calibrated uncertainty estimates for safer decision-making.

4. **Transferable Pretraining**  
   Demonstrates minimal-data fine-tuning on unseen cities to reduce data requirements.

5. **Practical Interpretability**  
   Generates intuitive risk maps for targetted urban safety interventions.

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt




### Extended Dependencies (for additional experiments and fine-tuning):
  ```bash
  pip install -r requirements-extended.txt

## Configuration

Create a `config.json` file in the root directory to set paths and hyperparameters. An example configuration structure:

```json
{
  "dataset_path": "Datasets/traffic_accident_data/",
  "model": "ALCo-FM",
  "training": {
    "epochs": 40,
    "batch_size": 8192,
    "learning_rate": 5e-5
  },
  "thresholds": {
    "tau_low": 0.2,
    "tau_high": 0.8
  },
  "save_dir": "results/",
  "seed": 42
}

```

Replace `"your_api_key_here"` and `"path/to/dataset"` with your actual token and dataset path.

---

## Project Structure

```plaintext
BTS/
├── codes/
│   ├── preprocessing_1.ipynb   # Data Preprocessing
│   ├── preprocessing_2.ipynb   # Data Preprocessing
│   ├── train.py                # Script to train the BTS model
│   ├── evaluate.py             # Script to evaluate model performance
│   ├── fine_tune.py            # Script for domain-adaptive fine-tuning on new cities
├── Dataset/                    # Scripts and utilities for data preprocessing and loading
├── requirements.txt            # Primary dependencies
├── requirements-extended.txt   # Extended dependencies for additional experiments
├── config.json                 # Configuration file for experiments
└── README.md                   # Project documentation
```

---

## Usage

### Preprocessing
Before training or fine-tuning, run the data preprocessing and analysis notebooks:

```bash
jupyter notebook code/preprocessing_1.ipynb
jupyter notebook code/preprocessing_2.ipynb
```


### Training

To train the ALCo-FM model, run:

```bash
python code/train.py --config config.json
```

### Evaluation

Evaluate the trained model using:

```bash
python code/evaluate.py --config config.json
```

### Fine-Tuning

For adapting the model to new cities or datasets:

```bash
python code/fine_tune.py --config config.json --city "NewCityName" --output_dir ./results/
```

---

## Evaluation Metrics

The framework is evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions.
- **F1 Score**: Harmonic mean of precision and recall.
- **Precision**: Ratio of true positives to all predicted positives.
- **Recall**: Ratio of true positives to all actual positives.
- **ECE** (Expected Calibration Error): Measures predictive calibration.

---

## Results

### Performance Tables

- **Table 1**: Ablation on Temporal Window Selection
- **Table 2**: Calibration Metrics (ECE, NLL)
- **Table 3**: SOTA Comparison on Training Cities
- **Table 4**: City-Wise Performance Metrics
- **Table 5**: Fine-Tuning Results on Unseen Cities

---

## Limitations and Future Work

### Limitations

- **Discrete Window Sizes**: Fixed 1 / 3 / 6 h bins may not capture all temporal dynamics.  
- **Calibration Overhead**: Monte Carlo dropout increases inference latency.  
- **Data Distribution Shift**: Performance can degrade under abrupt regime changes.  

### Future Work

- **Continuous Context Learning**: Train model to select arbitrary window lengths.  
- **Efficient Uncertainty Estimation**: Explore lightweight calibration methods.  
- **Additional Modalities**: Incorporate telematics, LiDAR, and mobile sensor data.  
- **Adaptive Thresholding**: Learn τ_low and τ_high per region or scenario.  


---

This repository includes the code, experiments, and supplementary materials for the **ALCo-FM** framework. For further details, please refer to the paper or contact the authors.

