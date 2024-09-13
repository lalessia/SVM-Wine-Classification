# Wine Classification using Support Vector Machines (SVM)

This project demonstrates the predictive analysis of a wine dataset to identify the producer, which serves as our target variable. The dataset is sourced from the sklearn library and contains the results of chemical analysis of wines grown in a specific region of Italy by three different producers.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Project Workflow](#project-workflow)
   - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Data Splitting and Normalization](#data-splitting-and-normalization)
   - [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
   - [Model Evaluation](#model-evaluation)
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
    - The parameters tuned include kernel type, regularization parameter `C`, `gamma`, and `degree` for polynomial kernels.
    
    ```python
    crossval = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    parameters = [
        {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
        {"kernel": ["rbf"], "C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]},
        {"kernel": ["poly"], "C": [0.01, 0.1, 1, 10, 100], "degree": np.arange(1, 5)}
    ]

    model = SVC()
    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=crossval, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Print best parameters
    print("Best Parameters:", clf.best_params_)
    ```

5. **Model Evaluation**
    - After training, the best model is evaluated on the test set.
    - Performance is measured using accuracy and the confusion matrix, which is also visualized for clarity.
    
    ```python
    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Model accuracy
    fitted_model = clf.best_estimator_
    predictions = fitted_model.predict(X_test)
    accuracy = round(accuracy_score(y_test, predictions), 4)
    print(f"Accuracy: {accuracy}")
    ```

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
