import gdown
import os

DATASET_URL = "https://drive.google.com/drive/folders/1C_mwQKEmwdWR3JISwgxtZsOepVCyJ9nR?usp=drive_link"
DATASET_DIR = 'dataset'

def download_dataset():
    dataset_dir = os.path.join(os.getcwd(), DATASET_DIR)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
        print(f"Folder created: {dataset_dir}")

    print("Downloading dataset...")
    gdown.download_folder(DATASET_URL, output=dataset_dir, quiet=False)
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_dataset()
