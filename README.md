# Beyond Time and Space: Multi-Modal Graph-Transformer Framework for Traffic Accident Prediction

A research paper and accompanying codebase presenting BTS—a unified multi-modal framework that integrates numerical, spatial, temporal, weather, and demographic data for accurate traffic accident risk prediction. The approach leverages transformer-based encoders, Vision Transformers (ViT), and Graph Neural Networks (GNN) to capture both local and global accident risk patterns.

---

## Overview

Beyond Time and Space (BTS) introduces a novel multi-modal Graph-Transformer framework that:
- **Integrates Heterogeneous Data**: Combines numerical time-series data, map-based visual representations, and spatio-temporal dependencies.
- **Adaptive Multi-Modal Fusion**: Utilizes transformer-based encoders for numerical features, ViT for spatial map images, and GNNs to model relationships across H3-based area nodes.
- **State-of-the-Art Performance**: Outperforms existing methods on key accident prediction metrics such as Accuracy, F1 Score, Precision, and Recall.
- **Interpretable Risk Mapping**: Outputs risk assessment maps that highlight safe (green) and high-risk (red) areas.

![Figure 1](images/fig1.png)   
*Figure 1: Overview of the proposed BTS framework. The model integrates numerical, spatial, temporal, weather, and demographic features using transformer-based encoders and a Vision Transformer (ViT) for spatial representation. A Graph Neural Network (GNN) captures spatio-temporal dependencies across H3-based area nodes, with a Multi-Layer Perceptron (MLP) head performing final accident risk classification. The framework outputs a risk assessment map, highlighting safe (green) and high-risk (red) areas.*

![Figure 2](images/fig2.png)  
*Figure 2: Comparison of state-of-the-art methods on accident-risk prediction across Accuracy, F1 Score, Precision, and Recall.*

---

## Key Contributions

1. **Multi-Modal Data Integration**: Seamlessly fuses numerical, visual, and spatial data for comprehensive accident risk assessment.
2. **Innovative Architecture**: Combines transformer-based encoders, Vision Transformers (ViT), and Graph Neural Networks (GNN) to capture local and global patterns.
3. **Adaptive Fusion Mechanism**: Dynamically balances contributions from each modality via an adaptive attention mechanism.
4. **Robust Performance**: Demonstrates superior accuracy and generalization across diverse urban environments.
5. **Interpretable Outputs**: Generates risk assessment maps to aid urban planners and policymakers in traffic safety interventions.

---

## Setup

### Dependencies

Install the required dependencies to reproduce the experiments:

- **Core Dependencies**:
  ```bash
  pip install -r requirements.txt



### Extended Dependencies (for additional experiments and fine-tuning):
  ```bash
  pip install -r requirements-extended.txt

## Configuration

Create a `config.json` file in the root directory to set paths and hyperparameters. An example configuration structure:

```json
{
  "dataset_path": "path/to/dataset",
  "model": "BTS",
  "training": {
    "epochs": 40,
    "batch_size": 8192,
    "learning_rate": 0.00005
  },
  "api_key": "your_api_key_here"
}
```

Replace `"your_api_key_here"` and `"path/to/dataset"` with your actual token and dataset path.

---

## Project Structure

```plaintext
BTS/
├── code/
│   ├── preprocessing_1.ipynb   # Data Preprocessing
│   ├── preprocessing_2.ipynb   # Data Preprocessing
│   ├── train.py                # Script to train the BTS model
│   ├── evaluate.py             # Script to evaluate model performance
│   ├── fine_tune.py            # Script for domain-adaptive fine-tuning on new cities
├── Datasets/                   # Scripts and utilities for data preprocessing and loading
├── images/                     # Figures and tables used in the paper
│   ├── fig1.png                # Overview of the proposed BTS framework
│   ├── fig2.png                # Comparison of state-of-the-art methods on accident-risk prediction
│   ├── tab1.png                # Evaluation of Transformer Models on Numerical Data
│   ├── tab2.png                # Evaluating Model’s Prediction After Incorporating Vision Transformers
│   ├── tab3.png                # Evaluating model’s prediction after integrating GNN
│   ├── tab4.png                # Evaluation Metrics Across Selected Cities Used for Training Data Collection in Accident Prediction
│   └── tab5.png                # Evaluation Metrics for Fine-Tuned Cities
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

To train the BTS model, run:

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

---

## Results

### Performance Tables

- **Table 1**: Evaluation of Transformer Models on Numerical Data  
- **Table 2**: Evaluating Model’s Prediction After Incorporating Vision Transformers  
- **Table 3**: Evaluating model’s prediction after integrating GNN  
- **Table 4**: Evaluation Metrics Across Selected Cities Used for Training Data Collection in Accident Prediction  
- **Table 5**: Evaluation Metrics for Fine-Tuned Cities  

---

## Limitations and Future Work

### Limitations

- **Scalability**: Increased computational costs for extremely large datasets.
- **Data Quality**: Model performance may vary with noisy or sparse input data.
- **Generalization**: Further improvements are needed to enhance adaptability in highly diverse urban scenarios.

### Future Work

- **Adaptive Spatial Indexing**: Refining spatial segmentation beyond fixed H3 grid sizes.
- **Integration of Additional Modalities**: Incorporating LiDAR, telematics, and mobile sensor data.
- **Model Optimization**: Exploring distributed training, model distillation, and pruning for real-time deployment.
- **Extended Benchmarks**: Testing on broader datasets to validate robustness across different regions.

---

This repository includes the code, experiments, and supplementary materials for the **BTS** framework. For further details, please refer to the paper or contact the authors.

