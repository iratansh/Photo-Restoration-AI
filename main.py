import torch
from PhotoRestorationLoss import PhotoRestorationLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FamilyPhotoRestorer import FamilyPhotoRestorer
from pathlib import Path

def train_network(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loss_weights = {
        'deblur': 1.0,
        'damage': 0.8,
        'color': 0.5,
        'face': 0.3
    }
    

    for stage in ['damage', 'deblur', 'color', 'face']:
        print(f"\n=== Training {stage} stage ===")
        optimizer = torch.optim.Adam(
            getattr(model, f"{stage}_net").parameters(),
            lr=0.0001,
            weight_decay=1e-5
        )
        
        for epoch in range(num_epochs//4):
            model.train()
            for batch in train_loader:
                degraded, clean = batch
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                output = model(degraded)
                loss = PhotoRestorationLoss()(output, clean) * loss_weights[stage]
                loss.backward()
                optimizer.step()

def process_directory(input_dir, output_dir, colorize=True, enhance_faces=True):
    restorer = FamilyPhotoRestorer()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for image_path in Path(input_dir).glob('*.[jJ][pP][gG]'):
        try:
            result = restorer.process_image(
                str(image_path),
                colorize=colorize,
                enhance_faces=enhance_faces
            )
            
            if result:
                output_path = Path(output_dir) / f'restored_{image_path.name}'
                result.save(output_path)
                print(f'Successfully processed {image_path.name}')
        except Exception as e:
            print(f'Error processing {image_path.name}: {str(e)}')

if __name__ == "__main__":
    input_dir = 'family_photos'
    output_dir = 'restored_photos'
    
    process_directory(
        input_dir,
        output_dir,
        colorize=True,
        enhance_faces=True
    )