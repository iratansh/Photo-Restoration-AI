from ConvBlock import ConvBlock
import torch.nn as nn
import torch, transforms, cv2, face_recognition
import numpy as np
from DeblurNetwork import DeblurNetwork
from ColorizationNetwork import ColorizationNetwork
from DamageRepairNetwork import DamageRepairNetwork
from FaceEnhancementNetwork import FaceEnhancementNetwork
from PIL import Image

class FamilyPhotoRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblur_net = self._add_spectral_norm(DeblurNetwork())
        self.colorizer = self._add_spectral_norm(ColorizationNetwork())
        self.damage_repair = self._add_spectral_norm(DamageRepairNetwork())
        self.face_enhancer = self._add_spectral_norm(FaceEnhancementNetwork())
        
    def _add_spectral_norm(self, module):
        return torch.nn.utils.spectral_norm(module)

    def forward(self, x):
        x = self._preprocess(x)
        x = self.deblur_net(x)
        x = self.damage_repair(x)
        x = self.colorizer(x)
        x = self.face_enhancer(x)
        
        return x

    def _preprocess(self, x):
        return (x - 0.5) * 2  
        
    def deblur_image(self, image):
        img_tensor = transforms.ToTensor()(image).unsqueeze(0)
        deblurred = self.deblur_net(img_tensor)
    
        return transforms.ToPILImage()(deblurred.squeeze(0))
    
    def colorize_bw(self, image):
        if len(np.array(image).shape) == 2 or np.array(image).shape[2] == 1:
            img_tensor = transforms.ToTensor()(image).unsqueeze(0)
            colorized = self.colorizer(img_tensor)
            return transforms.ToPILImage()(colorized.squeeze(0))
        return image
    
    def repair_damage(self, image):
        img_array = np.array(image)
        
        repaired = self.repair_tears(img_array)
        repaired = self.repair_creases(repaired)
        repaired = self.repair_stains(repaired)
        repaired = self.repair_scratches(repaired)
        
        return Image.fromarray(repaired)
    
    def repair_tears(self, img_array):
        edges = cv2.Canny(img_array, 100, 200)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return cv2.inpaint(img_array, dilated, 3, cv2.INPAINT_TELEA)
    
    def repair_creases(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
        return cv2.inpaint(img_array, mask, 3, cv2.INPAINT_NS)
    
    def enhance_faces(self, image):
        img_array = np.array(image)
        face_locations = face_recognition.face_locations(img_array)
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face = img_array[top:bottom, left:right]
            
            enhanced_face = self.face_enhancer(
                transforms.ToTensor()(Image.fromarray(face)).unsqueeze(0)
            )
            enhanced_face = transforms.ToPILImage()(enhanced_face.squeeze(0))
            img_array[top:bottom, left:right] = np.array(enhanced_face)
        
        return Image.fromarray(img_array)
        
    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        with torch.no_grad():
            repaired = self.damage_repair(img_tensor)
            deblurred = self.deblur_net(repaired)
            colorized = self.colorizer(deblurred)
            enhanced = self.face_enhancer(colorized)
            
        return transforms.ToPILImage()(enhanced.squeeze(0))

    def is_grayscale(self, image):
        img_array = np.array(image)
        if len(img_array.shape) < 3:
            return True
        if len(img_array.shape) == 3 and img_array.shape[2] == 1:
            return True
        return np.allclose(img_array[..., 0], img_array[..., 1]) and \
            np.allclose(img_array[..., 1], img_array[..., 2])
    
    def create_composite_reference(self, image_paths, output_path):
        images = [Image.open(path) for path in image_paths]
        
    
        aligned_images = []
        for img in images:
            aligned = self.align_image(img, images[0])
            aligned_images.append(aligned)
        
        masks = [self.detect_damage(img) for img in aligned_images]
        
        composite = np.zeros_like(np.array(images[0]))
        weight_sum = np.zeros_like(masks[0])
        
        for img, mask in zip(aligned_images, masks):
            img_array = np.array(img)
            inverse_mask = 1 - mask
            composite += img_array * inverse_mask[..., np.newaxis]
            weight_sum += inverse_mask
        
        composite = composite / (weight_sum[..., np.newaxis] + 1e-6)
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        Image.fromarray(composite).save(output_path)