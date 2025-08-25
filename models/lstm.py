import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMCell(nn.Module):
    """Custom LSTM Cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input gate, forget gate, cell gate, output gate
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = (torch.zeros(input.size(0), self.hidden_size, device=input.device, dtype=input.dtype),
                  torch.zeros(input.size(0), self.hidden_size, device=input.device, dtype=input.dtype))
        
        h_prev, c_prev = hx
        
        gates = (F.linear(input, self.weight_ih, self.bias_ih) + 
                F.linear(h_prev, self.weight_hh, self.bias_hh))
        
        # Split gates
        gates = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1])
        cell_gate = torch.tanh(gates[2])
        output_gate = torch.sigmoid(gates[3])
        
        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next


class LSTM(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
        output_size: int = 1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.output_size = output_size
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.lstm_layers.append(
                LSTMCell(layer_input_size, hidden_size)
            )
            if bidirectional:
                self.lstm_layers.append(
                    LSTMCell(layer_input_size, hidden_size)
                )
        
        # Output projection
        final_hidden_size = hidden_size * self.num_directions
        self.output_projection = nn.Linear(final_hidden_size, output_size)
        
        # Dropout layer
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass of LSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True
               or (seq_len, batch_size, input_size) if batch_first=False
            hx: Initial hidden state (h_0, c_0)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size) if batch_first=True
            or (seq_len, batch_size, output_size) if batch_first=False
        """
        if self.batch_first:
            # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states if not provided
        if hx is None:
            h = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                           device=x.device, dtype=x.dtype)
            c = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                           device=x.device, dtype=x.dtype)
        else:
            h, c = hx
        
        # Process through LSTM layers
        layer_input = x
        layer_idx = 0
        
        for layer in range(self.num_layers):
            layer_outputs = []
            
            if self.bidirectional:
                # Forward pass
                h_forward, c_forward = h[layer * 2], c[layer * 2]
                forward_outputs = []
                
                for t in range(seq_len):
                    h_forward, c_forward = self.lstm_layers[layer_idx](layer_input[t], (h_forward, c_forward))
                    forward_outputs.append(h_forward)
                    if self.dropout_layer and layer < self.num_layers - 1:
                        h_forward = self.dropout_layer(h_forward)
                
                # Backward pass
                h_backward, c_backward = h[layer * 2 + 1], c[layer * 2 + 1]
                backward_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    h_backward, c_backward = self.lstm_layers[layer_idx + 1](layer_input[t], (h_backward, c_backward))
                    backward_outputs.insert(0, h_backward)
                    if self.dropout_layer and layer < self.num_layers - 1:
                        h_backward = self.dropout_layer(h_backward)
                
                # Concatenate forward and backward outputs
                layer_output = torch.stack([torch.cat([f, b], dim=1) for f, b in zip(forward_outputs, backward_outputs)])
                layer_idx += 2
                
            else:
                # Unidirectional pass
                h_layer, c_layer = h[layer], c[layer]
                layer_outputs = []
                
                for t in range(seq_len):
                    h_layer, c_layer = self.lstm_layers[layer_idx](layer_input[t], (h_layer, c_layer))
                    layer_outputs.append(h_layer)
                    if self.dropout_layer and layer < self.num_layers - 1:
                        h_layer = self.dropout_layer(h_layer)
                
                layer_output = torch.stack(layer_outputs)
                layer_idx += 1
            
            layer_input = layer_output
        
        # Apply output projection
        output = self.output_projection(layer_input)
        
        if self.batch_first:
            # (seq_len, batch_size, output_size) -> (batch_size, seq_len, output_size)
            output = output.transpose(0, 1)
        
        return output


class LSTMForecaster(nn.Module):
    """Complete LSTM-based forecasting model"""
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        output_size: int = 1,
        use_tanh: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.use_tanh = use_tanh
        
        # LSTM backbone
        self.lstm = LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
            output_size=output_size
        )
        
        # Optional tanh activation for classification-like outputs
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
        output = self.lstm(x)
        
        if self.tanh is not None:
            output = self.tanh(output)
        
        return output

