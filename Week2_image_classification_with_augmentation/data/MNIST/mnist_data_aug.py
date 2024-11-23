import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomMNISTDataset_Aug(Dataset):

    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if self.train:
            self.image_file = os.path.join(self.root, "train-images-idx3-ubyte")
            self.label_file = os.path.join(self.root, "train-labels-idx1-ubyte")
        else:
            self.image_file = os.path.join(self.root, "t10k-images-idx3-ubyte")
            self.label_file = os.path.join(self.root, "t10k-labels-idx1-ubyte")

        if not os.path.exists(self.image_file) or not os.path.exists(self.label_file):
            print("MNIST Datasets are not existed.")

        self.images = self.read_data_file(self.image_file, image=True)
        self.labels = self.read_data_file(self.label_file, image=False)

    def read_data_file(self, path, image=True):
        with open(path, 'rb') as f:
            if image:
                data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def CustomMNIST_Aug(root, train=True):
    return CustomMNISTDataset_Aug(root=root, train=train)

