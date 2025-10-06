# Homework 2: Linear Regression and Support Vector Machines (SVM)

## Overview
This homework focuses on implementing and evaluating linear models for both regression and classification tasks. The assignment covers Support Vector Machines (SVM) implementation from scratch, linear regression techniques, and comprehensive model evaluation and comparison.

## Project Structure
- `ML_HM2_Q1.ipynb` - Linear Regression implementation and analysis
- `ML_HM2_Q2.ipynb` - Support Vector Machine (SVM) implementation and evaluation

## Question 1: Linear Regression Implementation

### Tasks Covered
1. **Dataset Analysis**
   - Comprehensive data exploration and visualization
   - Statistical analysis of features and target variables
   - Correlation analysis and feature relationships
   - Data quality assessment and preprocessing

2. **Linear Regression from Scratch**
   - Mathematical implementation of linear regression
   - Gradient descent optimization algorithm
   - Cost function implementation (Mean Squared Error)
   - Feature scaling and normalization techniques

3. **Model Training and Validation**
   - Train-test split implementation
   - Model fitting with different learning rates
   - Convergence analysis and optimization
   - Performance evaluation using multiple metrics

4. **Advanced Techniques**
   - Regularization methods (Ridge and Lasso regression)
   - Polynomial feature engineering
   - Cross-validation for model selection
   - Learning curve analysis

### Key Features
- **Custom Linear Regression**: Complete implementation from mathematical foundations
- **Gradient Descent**: Optimized parameter learning algorithm
- **Regularization Support**: Ridge and Lasso regression implementations
- **Comprehensive Evaluation**: Multiple regression metrics and visualizations
- **Feature Engineering**: Polynomial and interaction term creation

### Implementation Details
- **Cost Function**: Mean Squared Error with optional regularization terms
- **Optimization**: Batch and stochastic gradient descent variants
- **Convergence Criteria**: Automatic stopping based on improvement threshold
- **Visualization**: Learning curves, residual plots, and prediction accuracy

## Question 2: Support Vector Machine (SVM) Implementation

### Tasks Covered
1. **SVM Theory and Implementation**
   - Mathematical foundations of SVM optimization
   - Kernel trick implementation for non-linear classification
   - Support vector identification and margin calculation
   - Dual formulation and Lagrange multiplier optimization

2. **Kernel Functions**
   - Linear kernel implementation
   - Polynomial kernel with configurable degree
   - Radial Basis Function (RBF) kernel
   - Custom kernel function support

3. **Classification Tasks**
   - Binary classification implementation
   - Multi-class classification using one-vs-one and one-vs-all strategies
   - Decision boundary visualization
   - Margin analysis and support vector identification

4. **Hyperparameter Optimization**
   - Grid search for optimal C parameter (regularization)
   - Kernel parameter tuning (gamma for RBF, degree for polynomial)
   - Cross-validation for model selection
   - Performance comparison across parameter combinations

5. **Comparison with Scikit-learn**
   - Benchmarking custom implementation against sklearn
   - Performance and accuracy comparison
   - Computational efficiency analysis

### Key Features
- **Custom SVM Algorithm**: Complete implementation from optimization theory
- **Multiple Kernels**: Support for linear, polynomial, and RBF kernels
- **Margin Optimization**: Proper handling of support vectors and margins
- **Multi-class Support**: Extension to multi-class classification problems
- **Visualization Tools**: Decision boundaries and support vector plots

### Implementation Highlights
- **Quadratic Programming**: Optimization solver for SVM dual problem
- **Kernel Matrix**: Efficient computation and caching of kernel values
- **Support Vector Selection**: Identification of critical training examples
- **Prediction Pipeline**: Efficient classification for new data points

### Datasets Used
- **Synthetic Datasets**: Generated data for algorithm testing and visualization
- **Real-world Classification**: Standard benchmark datasets for evaluation
- **Linearly Separable**: Simple cases for algorithm validation
- **Non-linear Problems**: Complex datasets requiring kernel methods

## Learning Objectives
1. **Linear Model Theory**: Understand mathematical foundations of linear regression and SVM
2. **Optimization Algorithms**: Master gradient descent and quadratic programming techniques
3. **Kernel Methods**: Learn non-linear transformation techniques for classification
4. **Regularization**: Understand overfitting prevention and model generalization
5. **Performance Evaluation**: Master regression and classification evaluation metrics

## Technologies Used
- **Python**: Primary programming language
- **NumPy**: Numerical computations and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization and plotting
- **SciPy**: Optimization and statistical functions
- **Jupyter Notebooks**: Interactive development environment

## Mathematical Concepts
- **Linear Regression**: Least squares optimization and gradient descent
- **SVM Optimization**: Constrained optimization and Lagrange multipliers
- **Kernel Theory**: Feature space transformation and dot product computation
- **Regularization**: L1 and L2 penalty terms for model complexity control
- **Cross-validation**: Statistical model validation and selection

## Usage Examples
```python
# Linear Regression
lr = LinearRegression(learning_rate=0.01, max_iterations=1000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

# SVM Classification
svm = SVM(kernel='rbf', C=1.0, gamma=0.1)
svm.fit(X_train, y_train)
classifications = svm.predict(X_test)
```

## Results and Performance
- Custom linear regression achieves competitive performance with sklearn
- SVM implementation successfully handles both linear and non-linear classification
- Kernel methods significantly improve performance on complex datasets
- Hyperparameter optimization leads to substantial performance gains
- Regularization effectively prevents overfitting in both algorithms

## Advanced Topics
- **Feature Selection**: Automatic relevance detection in linear models
- **Ensemble Methods**: Combining multiple SVM classifiers
- **Online Learning**: Streaming data processing with gradient descent
- **Computational Optimization**: Efficient algorithms for large-scale problems