import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
import argparse
from typing import Tuple, Optional


def fetch_btc_data(
    symbol: str = "BTC-USD",
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    fetch historical BTC price data from Yahoo Finance.
    
    Args:
        symbol: Yahoo Finance symbol
        start_date: start date in YYYY-MM-DD format
        end_date: end date in YYYY-MM-DD format 
        interval: data interval
    
    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    
    # fetch data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for {symbol} in the specified date range")
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Select only the columns we need (OHLCV)
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    available_columns = list(data.columns)
    
    # Check which columns are available
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {available_columns}")
    
    # Select only available columns
    data = data[available_columns[:6]]  # Take first 6 columns (Date + OHLCV)
    
    # Ensure we have the right column names
    if len(data.columns) == 6:
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    else:
        print(f"Warning: Expected 6 columns, got {len(data.columns)}")
        print(f"Columns: {list(data.columns)}")
    
    print(f"Fetched {len(data)} records")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    return data


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets chronologically.
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Data split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/n:.1%})")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/n:.1%})")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df


def normalize_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'Close'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler fitted on training data.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        feature_columns: List of feature column names
        target_column: Target column name
    
    Returns:
        Tuple of normalized DataFrames and fitted scaler
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler on training data
    train_features = train_df[feature_columns].values
    scaler.fit(train_features)
    
    # Transform all datasets
    train_df_normalized = train_df.copy()
    val_df_normalized = val_df.copy()
    test_df_normalized = test_df.copy()
    
    train_df_normalized[feature_columns] = scaler.transform(train_features)
    val_df_normalized[feature_columns] = scaler.transform(val_df[feature_columns].values)
    test_df_normalized[feature_columns] = scaler.transform(test_df[feature_columns].values)
    
    print(f"Normalized {len(feature_columns)} features")
    print(f"Target column: {target_column}")
    
    return train_df_normalized, val_df_normalized, test_df_normalized, scaler


def save_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/btc_data",
    save_normalized: bool = True
):
    """
    Save datasets to CSV files.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Output directory
        save_normalized: Whether to save normalized data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    # Save data info
    info = {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'features': list(train_df.columns),
        'date_range': {
            'train': (train_df['Date'].min(), train_df['Date'].max()),
            'val': (val_df['Date'].min(), val_df['Date'].max()),
            'test': (test_df['Date'].min(), test_df['Date'].max())
        }
    }
    
    # Save info as text file
    with open(f"{output_dir}/data_info.txt", 'w') as f:
        f.write("BTC Data Information\n")
        f.write("===================\n\n")
        f.write(f"Total samples: {info['total_samples']}\n")
        f.write(f"Train samples: {info['train_samples']}\n")
        f.write(f"Validation samples: {info['val_samples']}\n")
        f.write(f"Test samples: {info['test_samples']}\n\n")
        f.write("Date ranges:\n")
        f.write(f"  Train: {info['date_range']['train'][0]} to {info['date_range']['train'][1]}\n")
        f.write(f"  Validation: {info['date_range']['val'][0]} to {info['date_range']['val'][1]}\n")
        f.write(f"  Test: {info['date_range']['test'][0]} to {info['date_range']['test'][1]}\n\n")
        f.write("Features:\n")
        for feature in info['features']:
            f.write(f"  - {feature}\n")
    
    print(f"Data saved to {output_dir}/")
    print(f"Files created:")
    print(f"  - train.csv ({len(train_df)} samples)")
    print(f"  - val.csv ({len(val_df)} samples)")
    print(f"  - test.csv ({len(test_df)} samples)")
    print(f"  - data_info.txt")


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC data and split into train/val/test sets")
    parser.add_argument("--symbol", default="BTC-USD", help="Yahoo Finance symbol")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--output-dir", default="data/btc_data", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--normalize", action="store_true", help="Apply normalization (disabled by default)")
    
    args = parser.parse_args()
    
    try:
        # Fetch data
        df = fetch_btc_data(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval
        )
        
        # Split data
        train_df, val_df, test_df = split_data(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Define feature columns (exclude Date and target)
        feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
        
        print(f"Features available: {feature_columns}")
        print(f"Number of features: {len(feature_columns)}")
        
        if args.normalize:
            # Normalize data
            train_df, val_df, test_df, scaler = normalize_data(
                train_df, val_df, test_df, feature_columns
            )
        else:
            print("Skipping normalization (raw data will be saved)")
        
        # Save data
        save_data(train_df, val_df, test_df, args.output_dir)
        
        print("\nData processing completed successfully!")
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
