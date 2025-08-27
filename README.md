# Deep Learning Approaches to Cryptocurrency Price Prediction

Leverage existed CryptoMamba structure with sentiment score and compare with other algorithms

Repository Includes:

* Implementation of CryptoMamba, LSTM, iTransformer, and Temporal Convolutional Network (TCN)
* Code for data preprocessing, model training, evaluation metrics

## Quick Start (modifications needed)


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
