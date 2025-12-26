"""Data loading and preprocessing with DVC integration."""

import pickle
from pathlib import Path
from typing import List, Tuple

import dvc.api


def download_data(data_dir: Path = Path("data")):
    """Download data from DVC remote storage or public sources.

    Args:
        data_dir: Directory to store data
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "crypto_data.pkl"

    try:
        with dvc.api.open("data/crypto_data.pkl", mode="rb") as f:
            content = pickle.load(f)
            output_path = data_dir / "crypto_data.pkl"
            with open(output_path, "wb") as out_f:
                pickle.dump(content, out_f)
        print(f"Data downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Error downloading data from DVC: {e}")
        print("Attempting to use local data if available...")


def get_price_list(content: dict, name_num: int = 0) -> Tuple[List[float], str]:
    """Obtain Price List.

    Args:
        content: .pkl file for crypto data
        name_num: 0 for BTC and 1 for ETH

    Returns:
        price_list: List of prices
        start_date: Starting date
    """
    price_list = []
    name_list = ["BTCBitcoin_price", "ETHEthereum_price"]
    desired = name_list[name_num]
    cnt = 0
    for name in content:
        if desired in name:
            if cnt == 0:
                start_date = name[0:8]
                cnt += 1
            price_list.append(content[name])
    return price_list, start_date


def load_crypto_data(
    data_path: Path, name_num: int = 0, train_split: float = 0.85
) -> Tuple[List[float], List[float], List[float], str]:
    """Load cryptocurrency data and split into train/test.

    Args:
        data_path: Path to data file
        name_num: 0 for BTC, 1 for ETH
        train_split: Train/test split ratio

    Returns:
        useful_data: All useful data
        train_data: Training data
        test_data: Test data
        start_date: Starting date
    """
    with open(data_path, "rb") as f:
        content = pickle.load(f)

    total_data, start_date = get_price_list(content, name_num=name_num)

    if name_num == 0:
        start_index = 2560
    else:
        start_index = 1800

    useful_data = total_data[start_index:]
    train_data = useful_data[0 : int(len(useful_data) * train_split)]
    test_data = useful_data[-int(len(useful_data) * (1 - train_split)) :]

    return useful_data, train_data, test_data, start_date


def prepare_data_for_training(
    data_path: Path, name_num: int = 0, train_split: float = 0.85, use_dvc: bool = True
) -> Tuple[List[float], List[float], List[float]]:
    """Prepare data for training with optional DVC download.

    Args:
        data_path: Path to data file
        name_num: 0 for BTC, 1 for ETH
        train_split: Train/test split ratio
        use_dvc: Whether to use DVC for data management

    Returns:
        useful_data: All useful data
        train_data: Training data
        test_data: Test data
    """
    if use_dvc and not data_path.exists():
        download_data(data_path.parent)

    useful_data, train_data, test_data, start_date = load_crypto_data(
        data_path, name_num, train_split
    )

    print(f"Data loaded from {start_date}")
    print(f"Total useful data: {len(useful_data)} days")
    print(f"Training data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")

    return useful_data, train_data, test_data
