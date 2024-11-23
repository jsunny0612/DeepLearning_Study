import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data_path = os.path.join(root, 'train' if train else 'test')
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to match VGG input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load images and labels
        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_path))):
            class_path = os.path.join(self.data_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def CustomCIFAR10(root, train=True):
    return CustomCIFAR10Dataset(root=root, train=train)



