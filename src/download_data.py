import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = [
    "mlg-ulb/creditcardfraud",
    "ealaxi/paysim1",
    "goyaladi/fraud-detection-dataset"
]

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def download_dataset(api: KaggleApi, dataset: str):
    print(f"Downloading {dataset} …")
    api.dataset_download_files(
        dataset, path="data", unzip=True, quiet=False
    )

def main():
    ensure_data_dir()

    api = KaggleApi()
    api.authenticate()

    for ds in DATASETS:
        download_dataset(api, ds)

    print("All datasets downloaded under ./data")

if __name__ == "__main__":
    main()