# Speech Emotion Recognition

A machine-learning pipeline for classifying emotions from speech recordings. The project extracts acoustic features from `.wav` files, combines MFCC and wavelet-based representations, and compares several classical machine-learning models for emotion classification.

> **Project status:** The repository currently contains the main feature-extraction and model-comparison notebook. The speech dataset and a standalone inference script are not included yet.

## Overview

Speech carries information beyond the words being spoken. Pitch, spectral structure, rhythm, and energy can provide signals about the speaker's emotional state. This project explores how engineered audio features can be used to classify short speech recordings into four emotion categories: **Neutral**, **Happy**, **Sad**, and **Angry**.

The workflow is intentionally transparent: audio is converted into numerical features, the features are standardized, classification models are trained, and their performance is compared using standard evaluation metrics.

## Main Workflow

```text
WAV recordings
      |
      v
Emotion-label extraction from filenames
      |
      v
MFCC + Wavelet feature extraction
      |
      v
Feature scaling and optional feature selection
      |
      v
Train/test split
      |
      v
SVM / Random Forest / MLP comparison
      |
      v
Accuracy, classification report, and confusion matrix
```

## Feature Engineering

### MFCC Features

The notebook loads up to three seconds from each audio file and calculates 40 Mel-Frequency Cepstral Coefficients. The mean value of each coefficient across the time dimension is used as a compact representation of the recording's spectral characteristics.

### Wavelet Features

A level-three discrete wavelet decomposition is calculated with the `db1` wavelet. For each coefficient group, the mean and standard deviation are collected. These features provide a complementary representation of local signal variation.

### Combined Representation

The MFCC vector and wavelet vector are concatenated into one feature vector. The resulting matrix is standardized with `StandardScaler`. The notebook also demonstrates `SelectKBest` with an ANOVA F-test to select the most informative features before classification.

## Models

| Model | Purpose |
|---|---|
| Support Vector Machine | Main non-linear classifier using an RBF kernel; grid search is included for hyperparameter tuning. |
| Random Forest | Ensemble baseline that captures non-linear relationships between audio features. |
| Multi-Layer Perceptron | Neural-network baseline for learning non-linear feature combinations. |

The evaluation section reports accuracy, a classification report, and a confusion matrix. The repository does not claim a final benchmark score because the dataset is not included and results depend on the exact recordings and split used.

## Dataset Format

https://drive.google.com/drive/folders/1QHm6rnklHEqF366CwvGV9Cx08R8Bq_61?usp=sharing

The current notebook expects a folder containing `.wav` files. It identifies emotion labels from the filename format. The implemented mapping is:

```python
emotion_dict = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
}
```

The notebook splits filenames using underscores and reads the emotion code from the third component. If your dataset uses a different filename convention, update the parsing logic before training.

A typical local layout is:

```text
project-root/
├── notebooks/
│   └── Feature Extraction.ipynb
├── dataset/
│   ├── speaker_001_ANG_001.wav
│   ├── speaker_002_HAP_002.wav
│   └── ...
└── README.md
```

The original notebook contains a Windows-specific dataset path. Change the `dataset_path` variable to the location of your own dataset before running the notebook.

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install numpy pandas librosa librosa-display matplotlib seaborn PyWavelets scikit-learn jupyter
```

If `librosa` reports an audio-backend issue, install the audio dependencies recommended by your operating system and restart the notebook kernel.

## Running the Project

1. Place the labeled `.wav` files in a local dataset folder.
2. Open `notebooks/Feature Extraction.ipynb` with Jupyter Notebook or JupyterLab.
3. Update `dataset_path` so it points to the local dataset folder.
4. Run the cells that discover the recordings and extract the labels.
5. Run the MFCC and wavelet feature-extraction cells.
6. Run the preprocessing and train/test split cells.
7. Train the SVM and the comparison models.
8. Review the accuracy, classification report, and confusion matrix.

Start Jupyter with:

```bash
jupyter notebook
```

## Recommended Improvements Before Production Use

The current notebook is an educational and experimental pipeline rather than a production inference service. A stronger version should add a reproducible `requirements.txt`, a standalone `train.py` script, a standalone `predict.py` script, model serialization, an explicit speaker-independent split, and a consistent feature-selection pipeline that is fitted only on the training data.

It is also important to evaluate the model with speaker-independent validation. Randomly splitting individual recordings can place recordings from the same speaker in both training and test sets, which may overestimate generalization performance.

## Limitations

- The dataset is not included in this repository.
- The notebook contains a local Windows path that must be changed.
- There is no standalone command-line or web inference application yet.
- The available labels cover four emotion classes only.
- Final performance depends strongly on recording quality, speaker distribution, language, and dataset balance.

## Future Work

Possible extensions include adding pitch and energy features, using log-mel spectrograms with CNN models, adding data augmentation such as noise and time shifting, performing speaker-independent cross-validation, exporting the trained pipeline with `joblib`, and exposing prediction through a FastAPI or Streamlit application.

## Repository Structure

```text
.
├── notebooks/
│   └── Feature Extraction.ipynb
├── .gitignore
└── README.md
```

## License and Dataset Notice

The project code is provided for learning and experimentation. Confirm the license and redistribution terms of any external speech dataset before sharing it or publishing derived recordings.
