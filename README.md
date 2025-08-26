# Deep Learning Approaches to Cryptocurrency Price Prediction

Additional approaches, including LSTM, iTransformer, and Temporal Convolutional Network (TCN), besides CryptoMamba for Bitcoin price prediction using State Space Models (SSMs)

## Features

- **CryptoMamba Model**: Custom Mamba-based architecture for time-series forecasting
- **Two Variants**: 
  - `cmamba_nv`: Without volume data (5 features: OHLC + Timestamp)
  - `cmamba_v`: With volume data (6 features: OHLCV + Timestamp)
- **Complete Pipeline**: Data preprocessing, training, evaluation, and next-day prediction
- **Trading Simulation**: Simple backtesting with buy/sell strategies

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train CryptoMamba without volume**:
   ```bash
   python scripts/training.py --config cmamba_nv --devices 1 --accelerator gpu --save_checkpoints
   ```

3. **Train CryptoMamba with volume**:
   ```bash
   python scripts/training.py --config cmamba_v --devices 1 --accelerator gpu --save_checkpoints
   ```

4. **Evaluate model**:
   ```bash
   python scripts/evaluation.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt
   ```

5. **Predict next day**:
   ```bash
   python scripts/one_day_pred.py --config cmamba_v --ckpt_path checkpoints/cmamba_v.ckpt --data_path data/one_day_pred.csv
   ```

## Configuration

- **Data Config**: `configs/data_configs/mode_1.yaml` - defines data source, time intervals, and splits
- **Training Configs**: 
  - `configs/training/cmamba_nv.yaml` - without volume
  - `configs/training/cmamba_v.yaml` - with volume
- **Model Configs**: 
  - `configs/models/CryptoMamba/v1.yaml` - 5 features, no normalization
  - `configs/models/CryptoMamba/v2.yaml` - 6 features, no normalization

## Data Format

Your raw data CSV should contain:
- `Timestamp` (or `Date` column)
- `Open`, `High`, `Low`, `Close` prices
- `Volume` (optional, for volume variant)

## Model Architecture

CryptoMamba uses:
- **Mamba SSM blocks** with selective state updates
- **Residual connections** and optional MLP branches
- **Configurable hidden dimensions** and layer density
- **Window-based processing** (default: 14 days)

## Outputs

- **Logs**: `logs/` (TensorBoard)
- **Checkpoints**: `checkpoints/` (best models)
- **Results**: `Results/<name>/<config>/` (evaluation plots)
- **Predictions**: `Predictions/<config>/<date>.txt` (next-day forecasts)

## Requirements

- PyTorch
- PyTorch Lightning
- mamba-ssm[causal-conv1d]
- pandas, numpy, matplotlib, seaborn
- See `requirements.txt` for full list

## Citation
```bibtex
@article{Sepehri2025CryptoMamba,
    title={CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction}, 
    author={Mohammad Shahab Sepehri and Asal Mehradfar and Mahdi Soltanolkotabi and Salman Avestimehr},
    year={2025},
    url={https://arxiv.org/abs/2501.01010}
}
```
