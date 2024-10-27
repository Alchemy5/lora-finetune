import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.image_paths = []
        self.transform = transform
        
        # Walk through the base directory and collect image file paths
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image
