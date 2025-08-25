# Time Series Forecasting Models

This directory contains three different neural network architectures for time series forecasting:

## 1. LSTM Model (`lstm.py`)

A Long Short-Term Memory (LSTM) network implementation for time series forecasting.

### Key Features:
- Custom LSTM cell implementation
- Support for bidirectional LSTM
- Multiple layers with dropout
- Configurable hidden size and number of layers

### Usage:
```python
from models.lstm import LSTMForecaster

# Create LSTM model
lstm_model = LSTMForecaster(
    num_features=5,          # Number of input features
    hidden_size=64,          # Hidden layer size
    num_layers=2,            # Number of LSTM layers
    dropout=0.1,             # Dropout rate
    bidirectional=False,     # Use bidirectional LSTM
    output_size=1,           # Output size (usually 1 for forecasting)
    use_tanh=False           # Apply tanh activation to output
)

# Forward pass
x = torch.randn(32, 100, 5)  # (batch_size, seq_len, num_features)
output = lstm_model(x)       # (batch_size, seq_len, output_size)
```

## 2. iTransformer Model (`itransformer.py`)

An inverted Transformer architecture designed specifically for time series forecasting.

### Key Features:
- Multi-head self-attention mechanism
- Positional encoding
- Feature embeddings
- Configurable transformer layers and dimensions

### Usage:
```python
from models.itransformer import iTransformerForecaster

# Create iTransformer model
itransformer_model = iTransformerForecaster(
    num_features=5,          # Number of input features
    d_model=512,             # Model dimension
    num_heads=8,             # Number of attention heads
    num_layers=6,            # Number of transformer layers
    d_ff=2048,               # Feed-forward dimension
    dropout=0.1,             # Dropout rate
    max_seq_len=5000,        # Maximum sequence length
    output_size=1,           # Output size
    use_tanh=False,          # Apply tanh activation
    use_feature_embedding=True  # Use feature embeddings
)

# Forward pass
x = torch.randn(32, 100, 5)  # (batch_size, seq_len, num_features)
output = itransformer_model(x)  # (batch_size, seq_len, output_size)
```

## 3. Temporal Convolutional Network (`tcn.py`)

A Temporal Convolutional Network using dilated causal convolutions for time series forecasting.

### Key Features:
- Dilated causal convolutions
- Residual connections
- Multiple temporal blocks
- Optional attention mechanism
- Configurable kernel sizes and dilations

### Usage:
```python
from models.tcn import TCNForecaster

# Create TCN model
tcn_model = TCNForecaster(
    num_features=5,          # Number of input features
    hidden_dims=[64, 128, 256],  # Hidden dimensions for each layer
    kernel_size=3,           # Convolution kernel size
    dropout=0.1,             # Dropout rate
    activation="relu",       # Activation function
    output_size=1,           # Output size
    use_tanh=False,          # Apply tanh activation
    use_attention=False      # Use attention mechanism
)

# Forward pass
x = torch.randn(32, 100, 5)  # (batch_size, seq_len, num_features)
output = tcn_model(x)        # (batch_size, seq_len, output_size)
```

## Model Comparison

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| **LSTM** | - Good at capturing long-term dependencies<br>- Simple and interpretable<br>- Works well with sequential data | - Can be slow to train<br>- May struggle with very long sequences<br>- Sequential processing | Medium-length sequences, when interpretability is important |
| **iTransformer** | - Excellent at capturing global dependencies<br>- Parallel processing<br>- State-of-the-art performance | - Higher computational cost<br>- More complex architecture<br>- Requires more data | Long sequences, when accuracy is paramount |
| **TCN** | - Fast training and inference<br>- Parallel processing<br>- Good at local patterns | - Limited receptive field<br>- May miss long-range dependencies<br>- More hyperparameters to tune | Real-time applications, when speed is important |

## Common Parameters

All models support these common parameters:

- `num_features`: Number of input features (e.g., OHLCV for financial data)
- `output_size`: Number of output features (usually 1 for price prediction)
- `use_tanh`: Whether to apply tanh activation to the output (useful for classification-like tasks)
- `dropout`: Dropout rate for regularization

## Input/Output Format

All models expect input in the format:
- **Input**: `(batch_size, sequence_length, num_features)`
- **Output**: `(batch_size, sequence_length, output_size)`

## Integration with Existing Code

These models can be easily integrated with the existing CryptoMamba framework by:

1. Importing the desired model
2. Creating an instance with appropriate parameters
3. Using it in place of the existing CMamba model

Example integration:
```python
# Replace CMamba with any of the new models
from models.lstm import LSTMForecaster
# from models.itransformer import iTransformerForecaster
# from models.tcn import TCNForecaster

model = LSTMForecaster(
    num_features=5,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    use_tanh=True  # For classification-like outputs
)
```

## Testing

Run the test script to verify all models work correctly:
```bash
python test_models.py
```

This will test all three models with sample data and verify their outputs are correct.

