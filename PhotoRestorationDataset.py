import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

class PhotoRestorationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', val_split=0.2):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.val_split = val_split
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
        self.clean_dir = self.data_dir / mode / 'clean'
        self.degraded_dir = self.data_dir / mode / 'degraded'
        
        if not self.clean_dir.exists():
            self.clean_dir.mkdir(parents=True, exist_ok=True)
        if not self.degraded_dir.exists():
            self.degraded_dir.mkdir(parents=True, exist_ok=True)
            
        self.image_pairs = self._load_image_pairs()
    
    def _load_image_pairs(self):
        """Load pairs of clean and degraded images"""
        pairs = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for img_path in self.clean_dir.glob('*.*'):
            if img_path.suffix.lower() in valid_extensions:
                degraded_path = self.degraded_dir / img_path.name
                if degraded_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        with Image.open(degraded_path) as img:
                            img.verify()
                        pairs.append((str(img_path), str(degraded_path)))
                    except Exception as e:
                        print(f"Skipping corrupt image pair: {img_path.name} - {str(e)}")
        
        print(f"Found {len(pairs)} valid image pairs in {self.mode} dataset")
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        clean_path, degraded_path = self.image_pairs[idx]
        
        clean_img = Image.open(clean_path).convert('RGB')
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            degraded_img = self.transform(degraded_img)
        
        return degraded_img, clean_img

    def prepare_dataset(self, source_dir):
        """
        Prepare dataset from source directory of original photos
        Args:
            source_dir: Path to directory containing original photos
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Get list of all images
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(list(source_dir.glob(ext)))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {source_dir}")
        
        # Split into train and validation sets
        train_files, val_files = train_test_split(
            image_files, 
            test_size=self.val_split, 
            random_state=42
        )
        
        # Process training set
        print("Processing training set...")
        self._process_image_set(train_files, 'train')
        
        # Process validation set
        print("Processing validation set...")
        self._process_image_set(val_files, 'val')
        
        print("Dataset preparation complete!")

    def _process_image_set(self, image_files, split):
        """Process a set of images for either training or validation"""
        for img_path in image_files:
            try:
                # Read original image
                img = Image.open(img_path).convert('RGB')
                
                # Save clean version
                clean_dir = self.data_dir / split / 'clean'
                clean_dir.mkdir(parents=True, exist_ok=True)
                clean_path = clean_dir / img_path.name
                img.save(clean_path)
                
                # Create degraded version
                degraded = self._create_degraded_version(img)
                
                # Save degraded version
                degraded_dir = self.data_dir / split / 'degraded'
                degraded_dir.mkdir(parents=True, exist_ok=True)
                degraded_path = degraded_dir / img_path.name
                degraded.save(degraded_path)
                
                print(f"Processed {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")

    def _create_degraded_version(self, img):
        """Create a degraded version of the input image"""
        img_array = np.array(img)
        
        # Apply random transformations
        img_array = self._add_noise(img_array)
        img_array = self._add_blur(img_array)
        img_array = self._add_scratches(img_array)
        img_array = self._add_fading(img_array)
        
        return Image.fromarray(img_array)

    def _add_noise(self, img_array):
        """Add random noise to image"""
        noise = np.random.normal(0, 25, img_array.shape)
        noisy_img = img_array + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def _add_blur(self, img_array):
        """Add Gaussian blur"""
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)

    def _add_scratches(self, img_array):
        """Add random scratches"""
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, img_array.shape[1])
            y1 = random.randint(0, img_array.shape[0])
            x2 = random.randint(0, img_array.shape[1])
            y2 = random.randint(0, img_array.shape[0])
            cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        return cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)

    def _add_fading(self, img_array):
        """Simulate aging/fading"""
        faded = img_array.astype(np.float32) * random.uniform(0.6, 0.9)
        return np.clip(faded, 0, 255).astype(np.uint8)

def main():
    dataset = PhotoRestorationDataset(
        data_dir='photo_restoration_data',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        mode='train'
    )
    
    # Prepare the dataset
    dataset.prepare_dataset(source_dir='old_photos')

if __name__ == "__main__":
    main()