# Fraud Detection System

This project focuses on building a robust system to detect fraudulent credit card transactions using machine learning techniques. The system preprocesses data, trains models like Autoencoder for anomaly detection, and deploys a scalable API for real-time fraud detection.

---

## Data

The data for the project can be found here - https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

## Key Features
1. **Data Preprocessing**:
  - Cleanses and normalizes transactional data for consistent input to models.
  - Handles imbalanced datasets using techniques like SMOTE.

2. **Model Development**:
  - Implements Autoencoder for anomaly detection.
  - Achieves high precision and recall to minimize false positives and false negatives.

3. **API Deployment**:
  - Provides a real-time fraud detection API built with **FastAPI**.
  - Designed for scalability and low-latency predictions.

4. **Visualization and Reporting**:
  - Generates insights into transaction trends and fraud patterns using **Matplotlib** and **Seaborn**.
  - Produces detailed evaluation metrics like ROC-AUC and confusion matrices.

---

## Project Structure
fraud_detection_project/
├── data/
│   ├── raw/                 # Raw dataset
│   └── processed/           # Processed dataset
├── models/
├── notebooks/
│   └── fraud_detection.ipynb   # Jupyter notebook for EDA, model training, and evaluation
├── reports/
├── src/
│   ├── preprocessing.py      # Data preprocessing and feature engineering
│   ├── model.py              # Model training and evaluation
│   └── visualization.py      # Functions for plotting and visualizations
├── api.py                    # FastAPI app for real-time predictions
├── requirements.txt          # List of dependencies
└── README.md                 # Project overview

## Setup Instructions
1. **Clone the Repository**
   git clone https://github.com/your_username/fraud_detection_project.git
   cd fraud_detection_project

2. **Install Dependencies**
   pip install -r requirements.txt

3. **Run the Jupyter Notebook**
  Open notebooks/fraud_detection.ipynb to explore the data and train the model.

4. **Start the API**
  Run the FastAPI app:
   uvicorn api:app --reload
  Access the API documentation at http://localhost:8000/docs.

## Usage
**Example API Request**
   curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
     "feature1": 0.56,
     "feature2": -1.23,
     "feature3": 2.45
   }'

**Example Response**
   {
     "fraudulent": true,
     "probability": 0.92
   }

## Technologies Used
- Python: Data processing and modeling.
- Scikit-learn: Machine learning algorithms.
- PyTorch: Deep learning for Autoencoders.
- FastAPI: API development and deployment.
- Matplotlib & Seaborn: Data visualization.
- Docker: Containerization for scalable deployment.

## Future Enhancements
- Implement more advanced models like XGBoost for improved performance.
- Add a user authentication layer for secure API usage.
- Integrate the system with cloud services for large-scale deployment.