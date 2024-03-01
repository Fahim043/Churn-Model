
# Churn Prediction Model

This project is aimed at predicting customer churn using various machine learning models including Random Forest, AdaBoost, Support Vector Machine (SVM), Multi-layer Perceptron (MLP), and XGBoost. The goal is to identify customers who are likely to churn based on historical data and various features.
## Data

The dataset used in this project contains historical records of customer interactions and whether they churned or not. It includes various features such as customer demographics, usage patterns, and interactions with the service. (Due to confidential issues the raw data can't be provided.)


## Approach

The project follows these key steps:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded and examined to understand its structure.
   - Null values are checked and handled as needed.
   - Categorical features are encoded using one-hot encoding if necessary.
   - Data is split into training and testing sets.
2. **Data Imbalance Handling**:
   - Since churn prediction is often an imbalanced classification problem, SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset.
   - Standard scaling is utilized to standardize the features by removing the mean and scaling to unit variance. This preprocessing step ensures that all features contribute equally to the model training process and prevents features with larger scales from dominating the learning process.
  By combining SMOTE for class imbalance handling and standard scaling for feature normalization, the project enhances the performance and robustness of the churn prediction models, resulting in more accurate and reliable predictions.
3. **Model Selection and Training**:
   - Several machine learning models including Random Forest, AdaBoost, SVM, MLP, and XGBoost are trained using pipelines.
   - Cross-validation is performed to evaluate the models using F1-score as the evaluation metric.
4. **Model Evaluation**:
   - The models are evaluated using various metrics such as F1-score, accuracy,and classification report.
   - Performance metrics are compared to select the best-performing model.


## **Results**:

- The models achieved high accuracy ranging from 88% to 94% on the test data. The F1-score, which considers both precision and recall, was used as the primary metric for evaluation.

## **Visualization**:
   - Visualizations such as classification reports, and feature importance plots may be created to interpret and communicate the results of the churn prediction models.




## Usage

To run this project locally, you can follow these steps:
 - Clone the repository to your local machine.
 - Ensure all dependencies are installed (NumPy, pandas, scikit-learn, imbalanced-learn, XGBoost, plotly).
 - Run the project's main script or Jupyter notebook to perform data analysis, model training, and evaluation.
## The project is organized as follows:

- `Churn_Model_Training.ipynb`: Jupyter notebook with code for data preprocessing, model training, and evaluation.
- `Churn testing.ipynb`: Jupyter notebooks with code for testing the trained model with new datasets for prediction.
- `README.md`: This file.


## Dependencies

The project utilizes the following Python libraries:
- NumPy
- pandas
- matplotlib
- imbalanced-learn
- XGBoost
- plotly
- scikit-learn

These dependencies can be easily installed using the `pip install` command.

## Author

This project was created by MD Fahim Afridi Ani. You can contact the author via Email: fahimafridi043@gmail.com.

Enjoy exploring.
