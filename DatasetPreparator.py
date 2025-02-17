import os
import cv2
import numpy as np
from pathlib import Path
import face_recognition
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import random

class DatasetPreparator:
    def __init__(self, source_dir, output_dir, val_split=0.2):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        
    def setup_directory_structure(self):
        for split in ['train', 'val']:
            for subdir in ['clean', 'degraded']:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def align_faces(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        face_locations = face_recognition.face_locations(image)
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        
        if not face_landmarks:
            return image
        
        try:
            left_eye = np.mean(face_landmarks[0]['left_eye'], axis=0)
            right_eye = np.mean(face_landmarks[0]['right_eye'], axis=0)
            

            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                        right_eye[0] - left_eye[0]))
            
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                         flags=cv2.INTER_CUBIC)
            
            return aligned_image
        except Exception as e:
            print(f"Warning: Could not align faces - {str(e)}")
            return image
    
    def detect_damage(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = cv2.Laplacian(blur, cv2.CV_64F).var()

        mask = np.zeros_like(gray)
        mask[edges > 0] = 1
        mask[texture < 100] = 1
    
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def create_composite(self, image_group):
        aligned_images = []
        masks = []
    
        reference = image_group[0]
        reference_array = np.array(Image.open(reference))
        
        for img_path in image_group:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            aligned = self.align_faces(img_array)
            
            mask = self.detect_damage(aligned)
            
            aligned_images.append(aligned)
            masks.append(mask)
        
        composite = np.zeros_like(aligned_images[0], dtype=np.float32)
        weights = np.zeros(aligned_images[0].shape[:2], dtype=np.float32)
        
        for img, mask in zip(aligned_images, masks):
            weight = 1 - mask
            composite += img.astype(np.float32) * weight[..., np.newaxis]
            weights += weight
    
        weights = np.maximum(weights, 1e-6)[..., np.newaxis]
        composite = composite / weights
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        return Image.fromarray(composite)
    
    def prepare_dataset(self):
        self.setup_directory_structure()
        
        image_groups = self.group_related_images()
        
        train_groups, val_groups = train_test_split(
            image_groups, test_size=self.val_split, random_state=42
        )
        
        print("Processing training data...")
        self._process_split(train_groups, 'train')
        print("Processing validation data...")
        self._process_split(val_groups, 'val')
    
    def group_related_images(self):
        image_files = list(self.source_dir.glob('*.jpg')) + \
                     list(self.source_dir.glob('*.jpeg')) + \
                     list(self.source_dir.glob('*.png'))
        
        groups = {}
        for img_path in image_files:
            group_key = img_path.stem[:8]
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(img_path)
        
        return list(groups.values())
    
    def _process_split(self, groups, split):
        for i, group in enumerate(groups):
            try:
                composite = self.create_composite(group)
                composite_path = self.output_dir / split / 'clean' / f'composite_{i}.jpg'
                composite.save(composite_path)
                
                for j, img_path in enumerate(group):
                    degraded_path = self.output_dir / split / 'degraded' / f'group_{i}_img_{j}.jpg'
                    shutil.copy(img_path, degraded_path)
                
                print(f"Processed group {i} in {split} split")
            except Exception as e:
                print(f"Error processing group {i}: {str(e)}")

def prepare_dataset(source_dir, output_dir, val_split=0.2):
    preparator = DatasetPreparator(source_dir, output_dir, val_split)
    preparator.prepare_dataset()