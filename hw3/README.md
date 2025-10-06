# Homework 3: Neural Networks and Multilayer Perceptrons

## Overview
This homework focuses on implementing and understanding neural networks, specifically multilayer perceptrons (MLPs) for both classification and regression tasks. The assignment covers neural network implementation from scratch, backpropagation algorithm, and comprehensive analysis of deep learning fundamentals.

## Project Structure
- `Q1.ipynb` - Complete neural network implementation and evaluation

## Neural Network Implementation

### Tasks Covered
1. **Feedforward Neural Network Architecture**
   - Multi-layer perceptron design and implementation
   - Configurable hidden layers and neuron counts
   - Activation function implementation (sigmoid, ReLU, tanh)
   - Weight initialization strategies (Xavier, He initialization)

2. **Backpropagation Algorithm**
   - Mathematical derivation and implementation of backpropagation
   - Gradient computation for weight updates
   - Chain rule application for multi-layer networks
   - Efficient matrix-based computations

3. **Training and Optimization**
   - Stochastic Gradient Descent (SGD) implementation
   - Mini-batch training for improved convergence
   - Learning rate scheduling and adaptive optimization
   - Early stopping and convergence criteria

4. **Data Preprocessing and Analysis**
   - Customer segmentation dataset preprocessing
   - Feature scaling and normalization techniques
   - Exploratory data analysis and visualization
   - Data cleaning and missing value handling

5. **Model Evaluation and Validation**
   - Cross-validation for model assessment
   - Performance metrics for classification and regression
   - Learning curve analysis and overfitting detection
   - Hyperparameter tuning and grid search

### Key Features
- **Custom Neural Network**: Complete implementation from mathematical foundations
- **Flexible Architecture**: Configurable layers, neurons, and activation functions
- **Backpropagation**: Efficient gradient computation and weight updates
- **Multiple Activation Functions**: Support for various non-linear transformations
- **Regularization Techniques**: Dropout and weight decay for overfitting prevention

### Network Architecture Components
1. **Input Layer**: Feature preprocessing and normalization
2. **Hidden Layers**: Configurable depth and width with activation functions
3. **Output Layer**: Task-specific output (classification/regression)
4. **Loss Functions**: Mean Squared Error, Cross-entropy, and custom losses
5. **Optimization**: Gradient descent variants and learning rate adaptation

### Implementation Details
- **Forward Pass**: Layer-by-layer computation with activation functions
- **Backward Pass**: Error propagation and gradient computation
- **Weight Updates**: Gradient descent with momentum and regularization
- **Batch Processing**: Efficient mini-batch training implementation

### Mathematical Foundations
- **Linear Transformations**: Matrix multiplications for layer computations
- **Activation Functions**: Non-linear transformations for feature learning
- **Loss Functions**: Objective optimization for training convergence
- **Gradient Computation**: Partial derivatives and chain rule application
- **Regularization**: L1/L2 penalty terms and dropout probability

### Dataset Analysis
- **Customer Data**: Behavioral and demographic features for segmentation
- **Feature Engineering**: Creation of relevant input features
- **Target Variables**: Multiple prediction tasks (classification/regression)
- **Data Visualization**: Comprehensive exploratory data analysis

## Advanced Neural Network Concepts

### Activation Functions
1. **Sigmoid**: Smooth gradient for binary classification
2. **ReLU**: Efficient computation and gradient flow
3. **Tanh**: Centered output for improved convergence
4. **Softmax**: Multi-class probability distributions

### Regularization Techniques
1. **Dropout**: Random neuron deactivation for generalization
2. **Weight Decay**: L2 regularization for parameter shrinkage
3. **Early Stopping**: Training termination based on validation performance
4. **Batch Normalization**: Internal covariate shift reduction

### Optimization Algorithms
1. **Stochastic Gradient Descent**: Basic optimization with momentum
2. **Adam Optimizer**: Adaptive learning rate with bias correction
3. **Learning Rate Scheduling**: Dynamic rate adjustment during training
4. **Gradient Clipping**: Prevention of exploding gradient problems

## Learning Objectives
1. **Neural Network Theory**: Understand mathematical foundations of deep learning
2. **Backpropagation**: Master gradient computation and error propagation
3. **Optimization**: Learn various training algorithms and convergence techniques
4. **Regularization**: Understand overfitting prevention and generalization
5. **Architecture Design**: Design effective network structures for different tasks

## Technologies Used
- **Python**: Primary programming language for implementation
- **NumPy**: Matrix operations and numerical computations
- **Pandas**: Data preprocessing and analysis
- **Matplotlib**: Visualization of training progress and results
- **Seaborn**: Statistical plotting and data exploration
- **Jupyter Notebook**: Interactive development and experimentation

## Implementation Highlights

### Network Class Structure
```python
class NeuralNetwork:
    def __init__(self, layers, activation='relu', learning_rate=0.01):
        # Network initialization
    
    def forward(self, X):
        # Forward propagation
    
    def backward(self, X, y):
        # Backpropagation algorithm
    
    def train(self, X, y, epochs, batch_size):
        # Training loop with optimization
    
    def predict(self, X):
        # Prediction on new data
```

### Training Pipeline
1. **Data Preparation**: Loading, cleaning, and preprocessing
2. **Network Architecture**: Layer configuration and initialization
3. **Training Loop**: Forward pass, loss computation, backpropagation
4. **Validation**: Performance monitoring and early stopping
5. **Evaluation**: Final model assessment and metrics

## Performance Analysis
- **Training Curves**: Loss and accuracy progression over epochs
- **Validation Performance**: Generalization assessment and overfitting detection
- **Hyperparameter Sensitivity**: Impact of learning rate, architecture, and regularization
- **Computational Efficiency**: Training time and memory usage analysis

## Results and Insights
- Custom neural network implementation achieves competitive performance
- Proper initialization and regularization significantly improve convergence
- Architecture design greatly impacts model capacity and generalization
- Backpropagation enables efficient learning in deep networks
- Optimization algorithm choice affects training stability and speed

## Advanced Extensions
- **Deep Networks**: Multi-layer architectures for complex pattern learning
- **Convolutional Layers**: Spatial feature extraction for image data
- **Recurrent Connections**: Sequential data processing capabilities
- **Attention Mechanisms**: Focus on relevant input features
- **Transfer Learning**: Pre-trained model adaptation for new tasks