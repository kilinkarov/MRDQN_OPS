"""Setup script to initialize data with DVC."""

from pathlib import Path


def main():
    """Initialize data directory and DVC tracking."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Data directory created.")
    print("\nTo add your data to DVC:")
    print("1. Place your crypto_data.pkl file in the data/ directory")
    print("2. Run: dvc add data/crypto_data.pkl")
    print("3. Run: git add data/crypto_data.pkl.dvc data/.gitignore")
    print("4. Run: git commit -m 'Add data tracking with DVC'")
    print("\nTo push data to remote:")
    print("dvc push")


if __name__ == "__main__":
    main()
