import pandas as pd
import os

def create_split_csv_files_with_paths(base_path, images_dir, labels_dir):
    """
    Create separate CSV files for test, train, and val splits with full paths
    
    Args:
        base_path (str): Base path where the text files are located
        images_dir (str): Directory containing images
        labels_dir (str): Directory containing labels/masks
    """
    # Define file paths relative to base path
    file_paths = {
        'test': os.path.join(base_path, 'test.txt'),
        'train': os.path.join(base_path, 'train.txt'), 
        'val': os.path.join(base_path, 'val.txt')
    }
    
    # Check if all files exist
    missing_files = []
    for split, file_path in file_paths.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Error: The following files are missing:")
        for missing_file in missing_files:
            print(f"  - {missing_file}")
        print("Please check the base path and try again.")
        return None
    
    print("All files found! Starting processing...")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    # Process each split file
    for split, file_path in file_paths.items():
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        missing_images = 0
        missing_labels = 0
        valid_files = 0
        
        for line_num, line in enumerate(lines, 1):
            filename = line.strip()
            if filename:  # Skip empty lines
                # Construct full paths (both have same filename)
                image_path = os.path.join(images_dir, filename)
                label_path = os.path.join(labels_dir, filename)  # Same filename for labels
                
                # Check if files exist
                image_exists = os.path.exists(image_path)
                label_exists = os.path.exists(label_path)
                
                if not image_exists:
                    missing_images += 1
                    print(f"  Warning: Image file not found: {image_path}")
                
                if not label_exists:
                    missing_labels += 1
                    print(f"  Warning: Label file not found: {label_path}")
                
                if image_exists and label_exists:
                    valid_files += 1
                    # Extract location and image number from filename
                    if '_' in filename and filename.endswith('.tif'):
                        location_part = filename[:-4]  # Remove .tif extension
                        parts = location_part.split('_')
                        
                        if len(parts) >= 2:
                            location = '_'.join(parts[:-1])  # Location name
                            image_number = parts[-1]         # Image number
                            
                            data.append({
                                'filename': filename,
                                'image_path': image_path,
                                'label_path': label_path,
                                'location': location,
                                'image_number': int(image_number),
                                'split': split
                            })
        
        # Create DataFrame for this split
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Sort by location and image number
            df = df.sort_values(['location', 'image_number']).reset_index(drop=True)
            
            # Save to CSV - only keep necessary columns for data loader
            output_file = f'{split}.csv'
            # Keep only the columns that your data loader needs
            csv_columns = ['image_path', 'label_path', 'filename']
            df[csv_columns].to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"  ✓ Created {output_file} with {len(df)} images")
            print(f"  ✓ Unique locations: {df['location'].nunique()}")
            print(f"  ✓ Valid files: {valid_files}")
            if missing_images > 0:
                print(f"  ⚠ Missing images: {missing_images}")
            if missing_labels > 0:
                print(f"  ⚠ Missing labels: {missing_labels}")
            
            # Show CSV structure
            print(f"  ✓ CSV columns: {csv_columns}")
        else:
            print(f"  ✗ No valid data found in {file_path}")
    
    return True

def verify_csv_structure():
    """Verify the structure of created CSV files"""
    print("\n" + "="*50)
    print("VERIFYING CSV STRUCTURE")
    print("="*50)
    
    for split in ['train', 'val', 'test']:
        csv_file = f'{split}.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"\n{split}.csv:")
            print(f"  Total records: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check first few records
            print("  First 3 records:")
            print(df.head(3)[['image_path', 'label_path', 'filename']].to_string(index=False))
            
            # Verify paths exist
            image_paths_exist = df['image_path'].apply(os.path.exists)
            label_paths_exist = df['label_path'].apply(os.path.exists)
            
            print(f"  Images exist: {image_paths_exist.sum()}/{len(df)}")
            print(f"  Labels exist: {label_paths_exist.sum()}/{len(df)}")

def test_data_loader_compatibility():
    """Test if the CSV files work with your data loader"""
    print("\n" + "="*50)
    print("TESTING DATA LOADER COMPATIBILITY")
    print("="*50)
    
    try:
        # Try to import and use your data loader
        import sys
        sys.path.append('.')  # Add current directory to path
        
        from data_loader import SatelliteDataset
        
        for split in ['train', 'val', 'test']:
            csv_file = f'{split}.csv'
            if os.path.exists(csv_file):
                print(f"\nTesting {split} dataset...")
                
                # Test with a simple transform
                from albumentations import Compose, Resize, Normalize
                from albumentations.pytorch import ToTensorV2
                
                simple_transform = Compose([
                    Resize(256, 256),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                
                dataset = SatelliteDataset(csv_file, transform=simple_transform)
                print(f"  ✓ Dataset loaded successfully: {len(dataset)} samples")
                
                # Test loading one sample
                if len(dataset) > 0:
                    image, mask, filename = dataset[0]
                    print(f"  ✓ Sample loaded: {filename}")
                    print(f"    Image shape: {image.shape}")
                    print(f"    Mask shape: {mask.shape}")
                    print(f"    Image type: {type(image)}")
                    print(f"    Mask dtype: {mask.dtype}")
                    print(f"    Unique mask values: {torch.unique(mask) if 'torch' in str(type(mask)) else 'N/A'}")
                    
    except ImportError as e:
        print(f"  ⚠ Could not import data_loader: {e}")
        print("  This is normal if you haven't implemented data_loader yet")
    except Exception as e:
        print(f"  ✗ Error testing data loader: {e}")

def create_dataset_summary():
    """Create a summary of the entire dataset"""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    summary_data = []
    
    for split in ['train', 'val', 'test']:
        csv_file = f'{split}.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Extract location from filename for summary
            df['location'] = df['filename'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else 'unknown')
            
            summary_data.append({
                'split': split,
                'total_images': len(df),
                'unique_locations': df['location'].nunique(),
                'locations_list': df['location'].unique().tolist()[:5]  # First 5 locations
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nDataset Overview:")
        for _, row in summary_df.iterrows():
            print(f"\n{row['split'].upper()} Split:")
            print(f"  Total images: {row['total_images']}")
            print(f"  Unique locations: {row['unique_locations']}")
            print(f"  Sample locations: {row['locations_list']}")
        
        # Save summary
        summary_df.to_csv('dataset_summary.csv', index=False)
        print(f"\n✓ Dataset summary saved to: dataset_summary.csv")

def main():
    """
    Main function to run the CSV creation process with paths
    """
    # Specify the base path where your text files are located
    base_path = r"K:\UNI\MAHBOD\OpenEarthMap\OpenEarthMap_wo_xBD"
    
    # Specify the directories containing images and labels
    images_dir = os.path.join(base_path, "images")
    labels_dir = os.path.join(base_path, "labels")
    
    print("Starting CSV creation process with paths...")
    print(f"Base path: {base_path}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print("-" * 50)
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    # Create CSV files with full paths
    success = create_split_csv_files_with_paths(base_path, images_dir, labels_dir)
    
    if success:
        # Verify the created CSV files structure
        verify_csv_structure()
        
        # Test data loader compatibility
        test_data_loader_compatibility()
        
        # Create dataset summary
        create_dataset_summary()
        
        print("\n" + "="*50)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nGenerated CSV files:")
        for split in ['train', 'val', 'test']:
            csv_file = f'{split}.csv'
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                print(f"  ✓ {csv_file}: {len(df)} images")
                print(f"    Columns: {list(df.columns)}")
                
        print("\nNow you can use these CSV files with your data loader!")
        print("\nExample usage in your code:")
        print("""
from data_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=8, 
    img_size=256
)
        """)
        
    else:
        print("\nProcess failed. Please check the file paths and try again.")

if __name__ == "__main__":
    main()