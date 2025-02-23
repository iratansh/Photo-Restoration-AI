import torch
from PhotoRestorationLoss import PhotoRestorationLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from FamilyPhotoRestorer import FamilyPhotoRestorer
from pathlib import Path
from PhotoRestorationDataset import PhotoRestorationDataset
from torchvision import transforms

def train_network(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loss_weights = {
        'deblur': 1.0,
        'damage_repair': 0.8,
        'colorizer': 0.5,
        'face_enhancer': 0.3
    }
    
    # Dictionary mapping stage names to network attributes
    network_mapping = {
        'deblur': 'deblur_net',
        'damage_repair': 'damage_repair',
        'colorizer': 'colorizer',
        'face_enhancer': 'face_enhancer'
    }

    for stage in network_mapping.keys():
        print(f"\n=== Training {stage} stage ===")
        optimizer = torch.optim.Adam(
            getattr(model, network_mapping[stage]).parameters(),
            lr=0.0001,
            weight_decay=1e-5
        )
        
        for epoch in range(num_epochs//4):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                degraded, clean = batch
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                output = model(degraded)
                loss = PhotoRestorationLoss()(output, clean) * loss_weights[stage]
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs//4}, Average Loss: {avg_loss:.4f}")

def process_directory(input_dir, output_dir, colorize=True, enhance_faces=True):
    restorer = FamilyPhotoRestorer()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for image_path in Path(input_dir).glob('*.[jJ][pP][gG]'):
        try:
            result = restorer.process_image(str(image_path))
            
            if result:
                output_path = Path(output_dir) / f'restored_{image_path.name}'
                result.save(output_path)
                print(f'Successfully processed {image_path.name}')
        except Exception as e:
            print(f'Error processing {image_path.name}: {str(e)}')

if __name__ == "__main__":
    # Create dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = PhotoRestorationDataset('photo_restoration_data', transform=transform, mode='train')
    val_dataset = PhotoRestorationDataset('photo_restoration_data', transform=transform, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Train model
    model = FamilyPhotoRestorer()
    train_network(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'family_photo_restorer.pth')

# === Training deblur stage ===
# Epoch 1/25, Average Loss: 0.5275
# Epoch 2/25, Average Loss: 0.5093
# Epoch 3/25, Average Loss: 0.4952
# Epoch 4/25, Average Loss: 0.4868
# Epoch 5/25, Average Loss: 0.4803
# Epoch 6/25, Average Loss: 0.4732
# Epoch 7/25, Average Loss: 0.4625
# Epoch 8/25, Average Loss: 0.4605
# Epoch 9/25, Average Loss: 0.4578
# Epoch 10/25, Average Loss: 0.4587
# Epoch 11/25, Average Loss: 0.4504
# Epoch 12/25, Average Loss: 0.4479
# Epoch 13/25, Average Loss: 0.4453
# Epoch 14/25, Average Loss: 0.4472
# Epoch 15/25, Average Loss: 0.4371
# Epoch 16/25, Average Loss: 0.4419
# Epoch 17/25, Average Loss: 0.4372
# Epoch 18/25, Average Loss: 0.4333
# Epoch 19/25, Average Loss: 0.4278
# Epoch 20/25, Average Loss: 0.4305
# Epoch 21/25, Average Loss: 0.4246
# Epoch 22/25, Average Loss: 0.4292
# Epoch 23/25, Average Loss: 0.4169
# Epoch 24/25, Average Loss: 0.4199
# Epoch 25/25, Average Loss: 0.4145

# === Training damage_repair stage ===
# Epoch 1/25, Average Loss: 0.3468
# Epoch 2/25, Average Loss: 0.3279
# Epoch 3/25, Average Loss: 0.3170
# Epoch 4/25, Average Loss: 0.3138
# Epoch 5/25, Average Loss: 0.3062
# Epoch 6/25, Average Loss: 0.3087
# Epoch 7/25, Average Loss: 0.3039
# Epoch 8/25, Average Loss: 0.2995
# Epoch 9/25, Average Loss: 0.3010
# Epoch 10/25, Average Loss: 0.3019
# Epoch 11/25, Average Loss: 0.3009
# Epoch 12/25, Average Loss: 0.2955
# Epoch 13/25, Average Loss: 0.2966
# Epoch 14/25, Average Loss: 0.2941
# Epoch 15/25, Average Loss: 0.2960
# Epoch 16/25, Average Loss: 0.2905
# Epoch 17/25, Average Loss: 0.2920
# Epoch 18/25, Average Loss: 0.2941
# Epoch 19/25, Average Loss: 0.2949
# Epoch 20/25, Average Loss: 0.2900
# Epoch 21/25, Average Loss: 0.2934
# Epoch 22/25, Average Loss: 0.2886
# Epoch 23/25, Average Loss: 0.2924
# Epoch 24/25, Average Loss: 0.2856
# Epoch 25/25, Average Loss: 0.2917

# === Training colorizer stage ===
# Epoch 1/25, Average Loss: 0.1818
# Epoch 2/25, Average Loss: 0.1808
# Epoch 3/25, Average Loss: 0.1777
# Epoch 4/25, Average Loss: 0.1810
# Epoch 5/25, Average Loss: 0.1814
# Epoch 6/25, Average Loss: 0.1784
# Epoch 7/25, Average Loss: 0.1785
# Epoch 8/25, Average Loss: 0.1789
# Epoch 9/25, Average Loss: 0.1782
# Epoch 10/25, Average Loss: 0.1776
# Epoch 11/25, Average Loss: 0.1780
# Epoch 12/25, Average Loss: 0.1779
# Epoch 13/25, Average Loss: 0.1788
# Epoch 14/25, Average Loss: 0.1756
# Epoch 15/25, Average Loss: 0.1770
# Epoch 16/25, Average Loss: 0.1766
# Epoch 17/25, Average Loss: 0.1742
# Epoch 18/25, Average Loss: 0.1767
# Epoch 19/25, Average Loss: 0.1783
# Epoch 20/25, Average Loss: 0.1762
# Epoch 21/25, Average Loss: 0.1733
# Epoch 22/25, Average Loss: 0.1772
# Epoch 23/25, Average Loss: 0.1775
# Epoch 24/25, Average Loss: 0.1762
# Epoch 25/25, Average Loss: 0.1736

# === Training face_enhancer stage ===
# Epoch 1/25, Average Loss: 0.0945
# Epoch 2/25, Average Loss: 0.0819
# Epoch 3/25, Average Loss: 0.0726
# Epoch 4/25, Average Loss: 0.0725
# Epoch 5/25, Average Loss: 0.0676
# Epoch 6/25, Average Loss: 0.0696
# Epoch 7/25, Average Loss: 0.0720
# Epoch 8/25, Average Loss: 0.0671
# Epoch 9/25, Average Loss: 0.0636
# Epoch 10/25, Average Loss: 0.0660
# Epoch 11/25, Average Loss: 0.0624
# Epoch 12/25, Average Loss: 0.0620
# Epoch 13/25, Average Loss: 0.0588
# Epoch 14/25, Average Loss: 0.0576
# Epoch 15/25, Average Loss: 0.0557
# Epoch 16/25, Average Loss: 0.0566
# Epoch 17/25, Average Loss: 0.0552
# Epoch 18/25, Average Loss: 0.0530
# Epoch 19/25, Average Loss: 0.0536
# Epoch 20/25, Average Loss: 0.0550
# Epoch 21/25, Average Loss: 0.0500
# Epoch 22/25, Average Loss: 0.0602
# Epoch 23/25, Average Loss: 0.0524
# Epoch 24/25, Average Loss: 0.0494
# Epoch 25/25, Average Loss: 0.0498