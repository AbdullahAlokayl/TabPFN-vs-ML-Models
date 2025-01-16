
---

# Comparing TabPFNClassifier with Other Machine Learning Models

This project evaluates the performance of the **TabPFNClassifier** in comparison to other popular machine learning models across multiple datasets. The evaluation focuses on metrics such as accuracy, fit time, and predict time to highlight the strengths and trade-offs of the TabPFNClassifier relative to conventional models.

## Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [How to Run](#how-to-run)
8. [License](#license)

---

## Introduction
The **TabPFNClassifier** is a tabular-specific pretrained feedforward neural network designed to deliver high accuracy on tabular data. This project benchmarks its performance against traditional machine learning models to identify scenarios where it excels and areas where it may face limitations.

---

## Objective
The primary goal of this project is to compare the performance of **TabPFNClassifier** with other popular models, including Logistic Regression, Gaussian Naive Bayes, k-Nearest Neighbors (k-NN), Decision Trees, and Support Vector Machines (SVM). The comparison evaluates:
- **Accuracy**: The ability to classify instances correctly.
- **Fit Time**: The time taken to train the model.
- **Predict Time**: The time required for making predictions.

---

## Datasets
The datasets used for evaluation are:
1. **Breast Cancer**: Predicts whether tumors are malignant or benign.
2. **Iris**: Classifies iris flowers into three species.
3. **Wine**: Categorizes wine into three classes based on chemical properties.

---

## Models
The models included in this study are:
1. **TabPFNClassifier** (Tabular Pretrained Feedforward Neural Network)
2. Logistic Regression
3. Gaussian Naive Bayes
4. k-Nearest Neighbors (k-NN)
5. Decision Tree
6. Support Vector Machine (SVM) with RBF kernel

---

## Results

| Dataset        | Model               | Accuracy  | Fit Time (ms) | Predict Time (ms) |
|----------------|---------------------|-----------|---------------|-------------------|
| breast_cancer  | LogisticRegression  | 0.978723  | 17.718315     | 0.000000          |
| breast_cancer  | GaussianNB          | 0.941489  | 0.995636      | 0.000000          |
| breast_cancer  | k-Nearest Neighbor  | 0.957447  | 0.000000      | 9.120703          |
| breast_cancer  | DecisionTree        | 0.920213  | 4.174471      | 0.000000          |
| breast_cancer  | SVM                 | 0.968085  | 2.934217      | 1.084566          |
| breast_cancer  | TabPFNClassifier    | 0.978723  | 0.000000      | 951.638937        |
| iris           | LogisticRegression  | 0.980000  | 4.997015      | 0.000000          |
| iris           | GaussianNB          | 0.960000  | 1.000643      | 0.000000          |
| iris           | k-Nearest Neighbor  | 0.980000  | 0.996351      | 1.993179          |
| iris           | DecisionTree        | 0.980000  | 0.000000      | 0.000000          |
| iris           | SVM                 | 0.980000  | 0.000000      | 0.000000          |
| iris           | TabPFNClassifier    | 0.980000  | 0.114679      | 223.053932        |
| wine           | LogisticRegression  | 0.983051  | 4.358292      | 0.000000          |
| wine           | GaussianNB          | 1.000000  | 0.000000      | 0.996113          |
| wine           | k-Nearest Neighbor  | 0.966102  | 0.000000      | 2.502203          |
| wine           | DecisionTree        | 0.966102  | 1.044512      | 0.000000          |
| wine           | SVM                 | 0.983051  | 0.741243      | 0.000000          |
| wine           | TabPFNClassifier    | 1.000000  | 0.998259      | 269.511700        |

---

## Conclusion

1. **TabPFNClassifier Strengths**:
   - Achieved the highest accuracy on all datasets, matching or exceeding other models.
   - Particularly effective on the Wine dataset, achieving perfect accuracy.

2. **TabPFNClassifier Limitations**:
   - The prediction time for the **TabPFNClassifier** is significantly longer than other models, especially on larger datasets like Breast Cancer (~951 ms).

3. **Efficiency Comparison**:
   - Models like **GaussianNB** and **DecisionTree** excel in computational efficiency, with minimal fit and prediction times.
   - Logistic Regression and SVM offer a good balance between accuracy and computational speed.

### Recommendations
- Use **TabPFNClassifier** for scenarios requiring top-tier accuracy, particularly when prediction speed is not a constraint.
- For real-time applications or larger datasets, consider faster models like **GaussianNB** or **DecisionTree**.

---

## How to Run

### Prerequisites
- Python 3.8 or later
- Required libraries: `scikit-learn`, `pandas`, `plotly`, `tabpfn`

### Installation
Install dependencies using pip:
```bash
pip install scikit-learn pandas plotly tabpfn
```

### Running the Notebook
Open the **main.ipynb** Jupyter Notebook and execute all cells to reproduce the analysis:
```bash
jupyter notebook main.ipynb
```

### Outputs
1. **Interactive Accuracy Graph**: Visualizes the accuracy of models across datasets.
2. **Interactive Time Graph**: Compares fit and predict times across models.
3. **Results DataFrame**: A tabular summary of metrics.

---

## License
This project is licensed under the MIT License.

---
