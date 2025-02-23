from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from FamilyPhotoRestorer import FamilyPhotoRestorer

def restore_old_photos(model_path, input_dir, output_dir):
    # Initialize model
    model = FamilyPhotoRestorer()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        for img_path in Path(input_dir).glob('*.[jJ][pP][gG]*'):
            try:
                # Load and preprocess
                image = Image.open(img_path).convert('RGB')
                tensor_img = transform(image).unsqueeze(0)
                
                # Run restoration
                output = model(tensor_img)
                
                # Save result
                result = transforms.ToPILImage()(output.squeeze(0))
                save_path = Path(output_dir) / f'restored_{img_path.name}'
                result.save(save_path)
                print(f"Restored {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    restore_old_photos(
        model_path='family_photo_restorer.pth',
        input_dir='old_photos',
        output_dir='restored_photos'
    )
