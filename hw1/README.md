# Homework 1: Decision Trees and K-Nearest Neighbors (KNN)

## Overview
This homework focuses on implementing and evaluating two fundamental machine learning algorithms: Decision Trees and K-Nearest Neighbors (KNN). The assignment includes both theoretical implementation from scratch and practical comparison with scikit-learn libraries.

## Project Structure
- `ml_hw1_q1_400243092_rashinrahnamoun.py` - Decision Tree implementation and evaluation
- `ml_hw1_q2_400243092_rashinrahnamoun.py` - KNN implementation and evaluation

## Question 1: Decision Tree Implementation

### Tasks Covered
1. **Data Preprocessing (Q1_a)**
   - Dataset exploration and analysis
   - Missing data handling using mean/mode imputation
   - Categorical encoding using label encoding
   - Feature normalization and standardization
   - Data preparation for machine learning models

2. **Decision Tree from Scratch (Q1_b)**
   - Custom implementation of Decision Tree classifier
   - Entropy and Gini impurity calculations
   - Information gain computation for feature selection
   - Tree building with configurable depth and minimum samples
   - Support for both numerical and categorical features

3. **Model Training and Evaluation (Q1_c)**
   - Custom train-test split implementation
   - Model training with hyperparameter tuning
   - Performance evaluation using accuracy, precision, recall, and F1-score
   - Multiple implementations with different splitting criteria

4. **Comparison with Scikit-learn (Q1_d)**
   - Performance comparison between custom implementation and sklearn
   - Evaluation of model consistency and accuracy

### Dataset
- **Loan Data**: Binary classification dataset for loan approval prediction
- Features include financial and demographic information
- Target variable: loan approval status

### Key Features
- **Custom Decision Tree Algorithm**: Complete implementation from scratch
- **Multiple Splitting Criteria**: Support for both entropy and Gini impurity
- **Flexible Tree Structure**: Configurable depth and minimum samples
- **Comprehensive Evaluation**: Multiple performance metrics
- **Data Processing Pipeline**: Complete preprocessing workflow

### Implementation Details
- **Tree Construction**: Recursive tree building with best split selection
- **Prediction Logic**: Tree traversal for classification
- **Evaluation Metrics**: Confusion matrix-based metrics calculation
- **Cross-validation**: Multiple training iterations for robust evaluation

### Usage
```python
# Load and preprocess data
df = pd.read_csv('loan_data.csv')
# ... preprocessing steps ...

# Train decision tree
dt = DecisionTree(max_depth=15, min_samples_split=2)
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)

# Evaluate performance
results = dt.evaluate(y_test, predictions)
```

## Question 2: K-Nearest Neighbors Implementation

### Tasks Covered
1. **Data Preprocessing**
   - Adult income dataset analysis and cleaning
   - Missing value treatment with interpolation and imputation
   - Feature encoding and selection
   - Exploratory data analysis with visualizations

2. **KNN from Scratch (Q2_b)**
   - Complete KNN classifier implementation
   - Euclidean distance computation
   - K-nearest neighbor selection
   - Majority voting for classification
   - Probability estimation for predictions

3. **Model Evaluation (Q2_c)**
   - Performance testing with different k values
   - Comprehensive evaluation metrics
   - Train-test split implementation

4. **Scikit-learn Comparison (Q2_d)**
   - Benchmarking against sklearn implementation
   - Performance and accuracy comparison

5. **Hyperparameter Optimization (Q2_e)**
   - Grid search for optimal parameters
   - Random search implementation
   - Cross-validation for model selection
   - Performance visualization

6. **Algorithm Efficiency (Q2_f)**
   - Comparison of different KNN algorithms (ball_tree, kd_tree, brute)
   - Time complexity analysis
   - Performance benchmarking

### Dataset
- **Adult Income Dataset**: Census data for income prediction
- Binary classification: <=50K vs >50K annual income
- Multiple demographic and employment features

### Key Features
- **Custom KNN Implementation**: Complete algorithm from scratch
- **Multiple Distance Metrics**: Support for different distance calculations
- **Hyperparameter Optimization**: Automated parameter tuning
- **Algorithm Comparison**: Performance analysis of different KNN variants
- **Comprehensive Evaluation**: Multiple metrics and visualizations

### Implementation Highlights
- **Efficient Neighbor Search**: Optimized distance computation
- **Flexible Architecture**: Configurable k values and metrics
- **Robust Evaluation**: Cross-validation and multiple test scenarios
- **Visualization Tools**: Performance plots and confusion matrices

### Usage
```python
# Train KNN classifier
knn = KNNClassifier(neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Evaluate model
metrics = knn.evaluate(X_test, y_test)
```

## Learning Objectives
1. **Algorithm Implementation**: Build machine learning algorithms from fundamental principles
2. **Data Preprocessing**: Master essential data preparation techniques
3. **Model Evaluation**: Understand comprehensive performance assessment
4. **Hyperparameter Tuning**: Learn optimization strategies for model performance
5. **Comparative Analysis**: Evaluate custom implementations against established libraries

## Technologies Used
- **Python**: Primary programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Benchmarking and comparison
- **Google Colab**: Development environment

## Results and Insights
- Custom implementations achieve comparable performance to scikit-learn
- Decision trees show good performance on structured data with proper preprocessing
- KNN performance is highly dependent on feature scaling and k selection
- Hyperparameter optimization significantly improves model performance
- Algorithm choice affects both accuracy and computational efficiency