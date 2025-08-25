import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder"""
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class iTransformer(nn.Module):
    """iTransformer model for time series forecasting"""
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        output_size: int = 1,
        use_tanh: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.output_size = output_size
        self.use_tanh = use_tanh
        
        # Input projection: project features to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Optional tanh activation
        if use_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Input projection: (batch_size, seq_len, num_features) -> (batch_size, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Optional tanh activation
        if self.tanh is not None:
            x = self.tanh(x)
        
        return x


class iTransformerForecaster(nn.Module):
    """Complete iTransformer-based forecasting model with additional features"""
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        output_size: int = 1,
        use_tanh: bool = False,
        use_feature_embedding: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.use_feature_embedding = use_feature_embedding
        self.use_tanh = use_tanh
        
        if use_feature_embedding:
            # Feature embedding for better representation
            self.feature_embedding = nn.Embedding(num_features, d_model)
            self.feature_projection = nn.Linear(d_model, d_model)
        
        # Main iTransformer model
        self.itransformer = iTransformer(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            output_size=output_size,
            use_tanh=False  # We'll handle tanh separately
        )
        
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
        if self.use_feature_embedding:
            # Create feature indices
            batch_size, seq_len, num_features = x.shape
            feature_indices = torch.arange(num_features, device=x.device).unsqueeze(0).unsqueeze(0)
            feature_indices = feature_indices.expand(batch_size, seq_len, -1)
            
            # Get feature embeddings
            feature_embeddings = self.feature_embedding(feature_indices)  # (batch_size, seq_len, num_features, d_model)
            feature_embeddings = self.feature_projection(feature_embeddings.mean(dim=2))  # (batch_size, seq_len, d_model)
            
            # Combine with original input
            x = x + feature_embeddings
        
        # Pass through iTransformer
        output = self.itransformer(x)
        
        # Apply tanh if needed
        if self.tanh is not None:
            output = self.tanh(output)
        
        return output

