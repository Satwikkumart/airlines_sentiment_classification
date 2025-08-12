✈️ **Airlines Sentiment Classification using NLP & MLOps**

This project leverages Natural Language Processing (NLP) techniques and MLOps best practices to build an end-to-end pipeline for classifying sentiment expressed in airline-related tweets. By fine-tuning a powerful pre-trained language model (DistilBERT), the system accurately predicts whether a tweet conveys a positive, neutral, or negative opinion about an airline. This project showcases a robust and reproducible machine learning workflow, from data acquisition to model deployment considerations.

✨ **Key Highlights**
**End-to-End ML Pipeline**: Demonstrates a complete machine learning lifecycle, including data ingestion, preprocessing, model training, evaluation, and considerations for deployment.

**State-of-the-Art NLP**: Utilizes DistilBERT, a fast and lightweight transformer model, achieving high accuracy in sentiment classification.

**MLOps Practices**: Implements experiment tracking, parameter management, and model versioning using MLflow, ensuring reproducibility and collaboration.

**Clear Performance Metrics**: Includes evaluation on a test set with relevant classification metrics (e.g., accuracy, precision, recall, F1-score).

**Modular and Scalable Design**: The codebase is organized into logical components, making it easy to understand, maintain, and potentially scale.


![sntim](https://github.com/user-attachments/assets/cf3b3425-39eb-4cbb-9ddf-6d25b095ed94)



📂 **Project Structure**
The project is organized as follows:

**artifacts/:** Stores intermediate and final outputs, such as processed data, trained models, and evaluation reports.

**config/:** Contains configuration files (config.yaml for pipeline settings and params.yaml for model hyperparameters).

**mlruns/:** MLflow's tracking store, containing logs of experiments, runs, and artifacts.

**research/:** Jupyter notebooks used for initial data exploration, experimentation, and prototyping.

**src/:** Contains the main source code organized into logical modules:

**airlinesSentiment/:** The core Python package.

**components/:** Implementations of each pipeline stage (data ingestion, preprocessing, model training, evaluation).

**config/:** Classes for managing project configurations.

**pipeline/:** Defines the execution workflow of the different stages.

**entity/:** Data classes and type definitions.

**utils/:** Utility functions and helper classes.

**main.py:** The entry point for running the entire ML pipeline.

**README.md:** This file, providing an overview of the project.

**requirements.txt:** Lists the Python dependencies required to run the project.

⚙️ **Getting Started**
Prerequisites

Python 3.8 or higher is required.

Ensure you have pip installed.

Basic understanding of machine learning concepts and the command line.

**Installation**

**Clone the repository:**

git clone https://github.com/Satwikkumart/airlines_sentiment_classification.git
cd airlines_sentiment_classification

**Create a virtual environment:**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install dependencies:**

pip install -r requirements.txt

Running the Pipeline

To execute the complete machine learning pipeline, simply run the main.py script:

python main.py

This will sequentially execute the defined stages: data ingestion, preprocessing, model training, and evaluation. Check the console output and the artifacts/ directory for progress and results.

Tracking Experiments with MLflow

To monitor the training process, compare different runs, and explore model parameters and metrics, start the MLflow UI:

**mlflow ui**

Then, open http://localhost:5000 in your web browser.

📊 **Performance Evaluation**
The trained DistilBERT model is evaluated on a held-out test dataset. Key performance metrics, such as accuracy, precision, recall, and F1-score, are calculated and logged using MLflow. These metrics provide insights into the model's ability to correctly classify the sentiment of unseen airline tweets. (Consider adding a specific evaluation result or a link to a report file in artifacts/)

