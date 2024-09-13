# Wine Classification using Support Vector Machines (SVM)

This project demonstrates the predictive analysis of a wine dataset to identify the producer, which serves as our target variable. The dataset is sourced from the sklearn library and contains the results of chemical analysis of wines grown in a specific region of Italy by three different producers.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Project Workflow](#project-workflow)
3. [Requirements](#requirements)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)
6. [Conclusion](#conclusion)

## Dataset Overview

The dataset consists of various chemical features of wine samples, with the target being the wine class (producer). More information about the dataset can be found [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset).

## Project Workflow

1. **Data Loading and Preprocessing**
    - The dataset is loaded from sklearn's `load_wine()` function and iit is then converted into a Pandas DataFrame for further analysis.

2. **Exploratory Data Analysis (EDA)**
    - We explore the dataset with functions like `.head()`, `.info()`, and `.describe()`.
    - We check for duplicate rows, class distribution, and investigate the relationships between features and the target variable using correlation matrices and visualizations.

3. **Data Splitting and Normalization**
    - The dataset is split into training and testing sets (70% train, 30% test) using stratified saipling to ensure balanced class distribution.

4. **Model Training and Hyperparameter Tuning**
    - We use a Support Vector Machine (SVM) classifier to predict the wine class.
    - A `GridSearchCV` is performed to optimize hyperparameters using cross-validation (`StratifiedKFold`) to ensure the model's robustness.
      
5. **Model Evaluation**
    - After training, the best model is evaluated on the test set.
    - Performance is measured using accuracy and the confusion matrix, which is also visualized for clarity.

## Requirements

To run this project, ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Acknowledgments

This project was developed as part of the Master in Data Science program at start2impact University. Special thanks to the team at start2impact for their guidance and support throughout the course. The program provided valuable insights and hands-on experience, enabling me to complete this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Conclusion

In this project, we built a classification model using SVMs to predict the wine producer based on the chemical composition of the wine. The model was optimized through grid search and cross-validation, and its performance was evaluated using accuracy and confusion matrix analysis.

