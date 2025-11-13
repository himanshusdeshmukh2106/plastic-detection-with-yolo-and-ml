"""
Helper script to check and verify dataset path
"""
import os
from pathlib import Path

def find_dataset():
    """Find the correct dataset path"""
    print("🔍 Searching for dataset...\n")
    
    possible_paths = [
        'training_dataset/training_dataset',
        'training_dataset',
        '../training_dataset/training_dataset',
        './training_dataset/training_dataset',
    ]
    
    for path in possible_paths:
        p = Path(path)
        if p.exists():
            # Check if it has class folders
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs:
                print(f"✅ Found dataset at: {p.absolute()}")
                print(f"   Classes found: {[d.name for d in subdirs]}")
                return str(p)
    
    print("❌ Dataset not found!")
    print("\n📦 Current directory contents:")
    for item in Path('.').iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")
    
    print("\n💡 To fix this:")
    print("   1. Make sure you're in the Plastic-Detection-Model directory")
    print("   2. Extract the dataset: unzip training_dataset.zip")
    print("   3. Verify the structure:")
    print("      training_dataset/")
    print("      └── training_dataset/")
    print("          ├── cardboard/")
    print("          ├── glass/")
    print("          ├── metal/")
    print("          ├── paper/")
    print("          ├── plastic/")
    print("          └── trash/")
    
    return None

def check_zip_file():
    """Check if zip file exists"""
    print("\n📦 Checking for zip file...")
    
    zip_paths = [
        'training_dataset.zip',
        '../training_dataset.zip',
    ]
    
    for zip_path in zip_paths:
        if Path(zip_path).exists():
            size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
            print(f"✅ Found zip file: {zip_path} ({size_mb:.1f} MB)")
            print(f"\n💡 Extract it with:")
            print(f"   unzip -q {zip_path}")
            return True
    
    print("❌ training_dataset.zip not found!")
    print("   Please download it from the repository")
    return False

if __name__ == '__main__':
    print("=" * 70)
    print("DATASET PATH CHECKER")
    print("=" * 70)
    
    # Check current directory
    print(f"\n📍 Current directory: {Path.cwd()}")
    
    # Find dataset
    dataset_path = find_dataset()
    
    if not dataset_path:
        # Check for zip file
        check_zip_file()
    else:
        print(f"\n✅ Dataset is ready!")
        print(f"   Use this path in your scripts: {dataset_path}")
    
    print("\n" + "=" * 70)
