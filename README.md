# ToyNet
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An educational but robust deep learning framework implemented from scratch using NumPy.  
Data loading is handled with Pandas.

ToyNet provides a clean, extensible architecture for building and training neural networks without external ML dependencies. Perfect for understanding deep learning mathematical fundamentals or experimenting with custom architectures on small to medium datasets.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Comprehensive Example](#comprehensive-example)
- [Mathematical Foundations](#mathematical-foundations)
- [Development](#development)
- [License](#license)

## Features
### **Core Capabilities**
- **Pure NumPy implementation** - No external ML libraries, transparent mathematical operations
- **Educational focus** - Clear, readable code designed for learning and understanding
- **Modular architecture** - Easily extensible with custom components
- **Type safety** - Full mypy compatibility with comprehensive type hints
- **Comprehensive tests** - Unit, integration, and end-to-end tests with high coverage

### **Data Handling**
- **Data loaders** - Support for CSV files and in-memory arrays
- **Preprocessing** - Programmable feature scaling, train/validation splits in data loaders
- **Batch processing** - Efficient mini-batch training with configurable sizes

### **Neural Network Components**
- **Layers**: Customizable with activation functions, weight initialization methods
- **Activation and loss functions**: Classic activations and losses for various tasks with the protocols to create custom ones
- **Optimizers**: Classic GD variants or Adam with adaptive learning rates

### **Training Features**
- **Training policies** - Early stopping, learning rate scheduling, model checkpointing, protocol for custom policies
- **Model persistence** - Save/load trained models in NumPy format
- **Logging** - Comprehensive training progress tracking

## Installation
### From PyPI (Recommended)
```bash
pip install toynet-ml
```

## Quick Start
### Simple XOR Problem
```python
import numpy as np
from toynet import MultiLayerPerceptron, Dense, BasicDataLoader
from toynet.functions import ReLU, Sigmoid, BinaryCrossEntropy
from toynet.optimizers import Adam

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 1, 2)
y = np.array([[0], [1], [1], [0]]).reshape(4, 1, 1)

# Build network
network = MultiLayerPerceptron(
    layers=[
        Dense(2, 8, ReLU),
        Dense(8, 1, Sigmoid)
    ],
    loss_function=BinaryCrossEntropy,
    optimizer=Adam(learning_rate=0.01)
)

# Train
data_loader = BasicDataLoader(X, y, batch_size=2)
network.train(data_loader, epochs=1000)

# Make batch predictions
predictions = network(X)
predictions_rounded = np.round(predictions, 2)
print(f"Input: \n{X.reshape(4, 2)}")
print(f"Predictions: {predictions_rounded.flatten()}")
```

### Simple Regression Problem: Predict y = 2x + 1
```python
import numpy as np
from toynet import MultiLayerPerceptron, Dense, BasicDataLoader
from toynet.functions import ReLU, Identity, MeanSquaredError
from toynet.optimizers import Adam

# Create regression dataset
X = np.array([[-4], [-3], [-2], [-1], [0], [1], [2], [3], [4], [5], [6]]).reshape(11, 1, 1)
y = np.array([[-7], [-5], [-3], [-1], [1], [3], [5], [7], [9], [11], [13]]).reshape(11, 1, 1)

# Build network
network = MultiLayerPerceptron(
    layers=[
        Dense(1, 8, ReLU),
        Dense(8, 8, ReLU),
        Dense(8, 1, Identity)
    ],
    loss_function=MeanSquaredError,
    optimizer=Adam(learning_rate=0.01)
)

# Train
data_loader = BasicDataLoader(X, y, batch_size=4)
network.train(data_loader, epochs=1000)

# Make unseen data prediction
prediction = network(np.array([9]).reshape(1, 1)) # 2x + 1 = 19
print(f"Predictions: {prediction.flatten()}") 
```

## Architecture Overview
ToyNet follows a modular, object-oriented design:

```
MultiLayerPerceptron
├── Layers (Dense)
│   ├── Input/Output dimensions
│   ├── Activation Functions (ReLU, Sigmoid, etc.)
│   └── Weight Initializers (He (default), Xavier)
├── Loss Function (CrossEntropy, MSE)
└── Optimizer (GD variants, Adam)

Training
├── Data Loader (Basic, CSV)
├── Fixed epochs
└── Training Policies (EarlyStopping, LR Scheduling etc.)
```

## Comprehensive Example
### End to end training on Kaggle Digit Recognizer CSV dataset
```python
import numpy as np

from toynet import Adam, CSVDataloader, Dense, MultiLayerPerceptron
from toynet.functions import (
    CategoricalCrossEntropy,
    ReLU,
    Softmax,
)
from toynet.policies import ReduceLROnPlateau, SaveBestModel, ValidationLossEarlyStop

if __name__ == "__main__":
    data_loader = CSVDataloader(
        "train.csv",
        batch_size=128,
        label_cols=["label"],
        validation_split=0.2,
        transform=lambda X, y: (X / 255.0, np.eye(10)[y.astype(int)]),
    )

    nnet = MultiLayerPerceptron(
        [
            Dense(784, 256, ReLU),
            Dense(256, 128, ReLU),
            Dense(128, 64, ReLU),
            Dense(64, 10, Softmax),
        ],
        loss_function=CategoricalCrossEntropy,
        optimizer=Adam(
            learning_rate=0.01,
        ),
    )

    nnet.train(
        data_loader,
        epochs=250,
        policies=[
            ValidationLossEarlyStop("mnist.npz", patience=6),
            ReduceLROnPlateau(factor=0.1, patience=4, min_lr=1e-6),
            SaveBestModel("mnist_best_checkpoint.npz", save_grace_period=20),
        ],
    )
```
### Benchmarks:
- Training time: ~40 minutes on modern CPU
- Best Kaggle test accuracy achieved: 96.6%

For production workloads, use PyTorch or TensorFlow which offer GPU acceleration and distributed training.


## Mathematical Foundations
ToyNet implements core neural network mathematics from scratch:

### Forward Propagation
```
h = σ(Wx + b)
```
Where:
- `W`: Weight matrix
- `x`: Input vector  
- `b`: Bias vector
- `σ`: Activation function

### Backpropagation
Automatic gradient computation using the chain rule:
```
∂L/∂W = ∂L/∂h × ∂h/∂z × ∂z/∂W
```
Gradients are computed and stored in layer objects during backpropagation.

### Weight Updates
- **GD variants**: `W ← W - η∇W`
- **Adam**: Adaptive learning with momentum and RMSprop

## Development
### Running tests and code checks
After cloning the repository and setting up a virtual environment:
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest --cov=toynet --cov-report=html

# Type checking
mypy --install-types --config-file pyproject.toml ./src

# Code formatting
ruff format
ruff check --fix
```

### Project Structure
```
toynet/
├── src/toynet/              # Main package
│   ├── data_loaders/        # Data loading utilities
│   ├── functions/           # Activation and loss functions
│   ├── initializers/        # Weight initialization
│   ├── layers/              # Layer implementations
│   ├── networks/            # Neural network architectures
│   ├── optimizers/          # Gradient descent algorithms
│   ├── policies/            # Training policies
│   └── config.py            # Configuration settings
│
├── tests/                   # Test suite
│   ├── uts/                 # Unit tests
│   ├── integration/         # Integration tests
└───└── e2e/                 # End-to-end tests
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
