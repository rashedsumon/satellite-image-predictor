import os
import kagglehub

# List of the 10 target classes in the exact order expected by the dataset structure
CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", 
    "Industrial", "Pasture", "Permanent Crop", "Residential", 
    "River", "Sea/Lake"
]

def download_dataset():
    """
    Downloads the latest version of the EuroSAT dataset using kagglehub.
    Returns the absolute local path to the dataset folder.
    """
    print("Checking/Downloading EuroSAT dataset via kagglehub...")
    # Download latest version
    path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
    print("Path to dataset files:", path)
    
    # The actual images are typically inside a subfolder named '2750' or 'EuroSAT'
    # We will look for '2750' which contains the 10 class directories
    potential_data_dir = os.path.join(path, "EuroSAT", "2750")
    if os.path.exists(potential_data_dir):
        return potential_data_dir
        
    potential_data_dir_alt = os.path.join(path, "2750")
    if os.path.exists(potential_data_dir_alt):
        return potential_data_dir_alt
        
    return path

if __name__ == "__main__":
    # Test script execution
    dataset_path = download_dataset()
    print(f"Data ready at: {dataset_path}")