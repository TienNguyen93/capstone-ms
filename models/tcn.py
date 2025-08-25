import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class Chomp1d(nn.Module):
    """Remove extra elements from the right side of the output"""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal Block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dropout = dropout
        
        # Calculate padding to maintain sequence length
        if padding == 0:
            padding = (kernel_size - 1) * dilation
        
        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.downsample is not None:
            x = self.downsample(x)
        
        return self.activation(out + x)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, num_channels[-1])
        """
        # Transpose to (batch_size, num_features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.network(x)
        # Transpose back to (batch_size, seq_len, num_channels[-1])
        x = x.transpose(1, 2)
        return x


class TCN(nn.Module):
    """Complete TCN model for time series forecasting"""
    
    def __init__(
        self,
        num_features: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        output_size: int = 1,
        use_tanh: bool = False
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.output_size = output_size
        self.use_tanh = use_tanh
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        
        # Output projection
        self.output_projection = nn.Linear(num_channels[-1], output_size)
        
        # Optional tanh activation
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Pass through TCN
        x = self.tcn(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Optional tanh activation
        if self.tanh is not None:
            x = self.tanh(x)
        
        return x


class TCNForecaster(nn.Module):
    """Complete TCN-based forecasting model with additional features"""
    
    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        output_size: int = 1,
        use_tanh: bool = False,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.output_size = output_size
        self.use_tanh = use_tanh
        self.use_attention = use_attention
        
        # TCN backbone
        self.tcn = TCN(
            num_features=num_features,
            num_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            output_size=hidden_dims[-1],
            use_tanh=False
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dims[-1], output_size)
        
        # Optional tanh activation
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Pass through TCN
        x = self.tcn(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_output, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attn_output)
        
        # Output projection
        x = self.output_projection(x)
        
        # Optional tanh activation
        if self.tanh is not None:
            x = self.tanh(x)
        
        return x


class DilatedCausalConv1d(nn.Module):
    """Dilated Causal Convolution 1D"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution
        x = self.conv(x)
        # Remove extra padding from the right
        x = x[:, :, :-(self.kernel_size - 1) * self.dilation]
        return x


class ResidualBlock(nn.Module):
    """Residual Block with dilated causal convolution"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        
        # First dilated causal convolution
        self.conv1 = DilatedCausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        
        # Second dilated causal convolution
        self.conv2 = DilatedCausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.downsample is not None:
            x = self.downsample(x)
        
        return self.activation(out + x)

