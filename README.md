

# Brain Tumor Detection System

An AI-powered web application that classifies brain MRI images into four categories using a deep learning model and provides predictions through an interactive Streamlit interface.

## Overview

The Brain Tumor Detection System uses a trained Xception-based image classification model to analyze uploaded brain MRI images and predict one of four classes. The application provides a simple interface for uploading an image and viewing the predicted class with its associated confidence score.

**This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.**

## Features

- Upload brain MRI images in JPG, JPEG, and PNG formats.
- Classify images into four categories.
- Display the predicted category and class probabilities.
- Interactive web interface built with Streamlit.
- Transfer learning using the Xception architecture.
- Saved model for inference without retraining.

## Classification Categories

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary Tumor

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architecture:** Xception with transfer learning
- **Web Framework:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Version Control:** Git and GitHub

## Project Structure

```text
Brain-Tumor-Detection-System/
├── Training/                  # Training dataset (local only)
├── Testing/                   # Testing dataset (local only)
├── app.py                     # Streamlit web application
├── train_model.py             # Model training script
├── braintumorclassification.py # Original classification script
├── brain_tumor_model.keras    # Trained model
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignored files and folders
└── README.md                  # Project documentation
```

The `Training/` and `Testing/` folders are excluded from GitHub through `.gitignore` to avoid uploading the large image datasets.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jayasen3933/Brain-Tumor-Detection-System.git
cd Brain-Tumor-Detection-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Open it in your browser to use the application.

## Model Training

The `train_model.py` script trains the image classification model using the local training dataset and saves the trained model as `brain_tumor_model.keras`.

To train the model again, ensure the dataset folders are available locally and run:

```bash
python train_model.py
```

Training requires additional time and memory. You can use the existing saved model to run the application without retraining.

## Model Evaluation

The current training run reported the following results:

- **Training accuracy:** 89.66%
- **Validation accuracy:** 89.52%
- **Test accuracy:** 88.25%

These are results from the current experiment, not a guarantee of real-world performance. Independent validation, class-wise evaluation, and checks for data leakage are necessary before drawing stronger conclusions.

## Future Improvements

- Add class-wise precision, recall, F1-score, and a confusion matrix.
- Improve model evaluation and robustness across different MRI sources.
- Enhance the user interface and deployment experience.
- Explore model interpretability techniques such as Grad-CAM.

## Disclaimer

This application is a student project for learning and demonstration. Predictions may be incorrect and must not be used to make medical decisions. Consult a qualified healthcare professional for medical evaluation.

## Author

**Chada Jayasen Reddy**

- GitHub: [jayasen3933](https://github.com/jayasen3933)

## License

No license has been specified for this repository yet. All rights are reserved unless a license is added.