import kagglehub
import os
import shutil

# Path to the dataset folder within the project
dataset_dir = os.path.join(os.getcwd(), 'stock_data')

# Create the dataset folder if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Folder created: {dataset_dir}")

# Download the dataset
path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
print("Path to dataset files:", path)

# Move all downloaded files to the dataset folder
for file in os.listdir(path):
    source_file = os.path.join(path, file)
    dest_file = os.path.join(dataset_dir, file)
    shutil.move(source_file, dest_file)
    print(f"File moved: {file} → {dest_file}")

print(f"All files have been moved to {dataset_dir}")