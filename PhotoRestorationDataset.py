import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import random
from FamilyPhotoRestorer import FamilyPhotoRestorer
from PhotoRestorationLoss import PhotoRestorationLoss
from sklearn.model_selection import train_test_split

class PhotoRestorationDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
        self.clean_dir = self.data_dir / 'clean'
        self.degraded_dir = self.data_dir / 'degraded'
        
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean images directory not found: {self.clean_dir}")
        if not self.degraded_dir.exists():
            raise FileNotFoundError(f"Degraded images directory not found: {self.degraded_dir}")
            
        self.image_pairs = self._load_image_pairs()
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No valid image pairs found in {self.data_dir}")
    
    def _load_image_pairs(self):
        pairs = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for img_path in self.clean_dir.iterdir():
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
                        print(f"Warning: Skipping corrupt image pair: {img_path.name} - {str(e)}")
        
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

    def _apply_random_degradation(self, img):
        img_array = np.array(img)
        
        # Add multiple damage types
        img_array = self._add_scratches(img_array)
        img_array = self._add_fading(img_array)
        img_array = self._add_stains(img_array)
        
        return Image.fromarray(img_array)
    
    def _add_scratches(self, img_array):
        """Add realistic scratches to image"""
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        for _ in range(random.randint(1, 3)):
            length = random.randint(20, 100)
            x = random.randint(0, img_array.shape[1]-1)
            y = random.randint(0, img_array.shape[0]-1)
            cv2.line(mask, (x,y), (x+length,y), 255, thickness=random.randint(1,2))
        return cv2.inpaint(img_array, mask, 3, cv2.INPAINT_NS)

    def _add_fading(self, img_array):
        """Simulate color fading"""
        faded = img_array.astype(np.float32) * random.uniform(0.5, 0.8)
        faded += random.randint(10, 30) 
        return np.clip(faded, 0, 255).astype(np.uint8)

    def _add_stains(self, img_array):
        """Add coffee-like stains"""
        stains = np.zeros_like(img_array)
        for _ in range(random.randint(1, 2)):
            x = random.randint(0, stains.shape[1]-50)
            y = random.randint(0, stains.shape[0]-50)
            stains[y:y+50, x:x+50] = random.randint(100, 200)
        return cv2.addWeighted(img_array, 0.8, stains, 0.2, 0)
    
    def prepare_dataset(self):
        """Prepare the complete dataset"""
        self.setup_directory_structure()
        
        # Group related images (photos from same time/event)
        image_groups = self.group_related_images()
        
        train_groups, val_groups = train_test_split(
            image_groups, test_size=self.val_split, random_state=42
        )
        
        print("Processing training data...")
        self._process_split(train_groups, 'train')
        
        print("Processing validation data...")
        self._process_split(val_groups, 'val')

def main():
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'image_size': 256
    }
    config['image_size'] = 128 
    config['num_epochs'] = 5 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    data_dir = Path('photo_restoration_data')
    
    try:
        print("Loading training dataset...")
        train_dataset = PhotoRestorationDataset(
            data_dir / 'train',
            transform=transform,
            mode='train'
        )
        
        print("Loading validation dataset...")
        val_dataset = PhotoRestorationDataset(
            data_dir / 'val',
            transform=transform,
            mode='val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Successfully created data loaders:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
 
        model = FamilyPhotoRestorer()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = PhotoRestorationLoss()
        
        for epoch in range(config['num_epochs']):
            model.train()
            for degraded, clean in train_loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                output = model(degraded)
                loss = criterion(output, clean)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for degraded, clean in val_loader:
                    output = model(degraded.to(device))
                    val_loss += criterion(output, clean.to(device)).item()
        print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Preparing dataset...")
    photoDataset = PhotoRestorationDataset(
        data_dir='photo_restoration_data',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        mode='train'
    )
    photoDataset.prepare_dataset(
        source_dir='old_photos',
        output_dir='photo_restoration_data',
        val_split=0.2  
    )
    # main()