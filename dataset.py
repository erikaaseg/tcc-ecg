import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os


class ECGDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 crop: tuple[int, int, int, int] | None = None):
        """
        Args:
            root_dir (str): Directory with all the ECG images.
            crop (tuple[int, int, int, int], optional): Crop rectangle as (left, right, upper, lower). Defaults to None.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels: list[int] = []

        self.classes = ['MI', 'PMI', 'HB', 'Normal']
        classes_dirs = [
            'ECG Images of Myocardial Infarction Patients (240x12=2880)',
            'ECG Images of Patient that have History of MI (172x12=2064)',
            'ECG Images of Patient that have abnormal heartbeat (233x12=2796)',
            'Normal Person ECG Images (284x12=3408)'
        ]
        # find all images and their labels running through the directories
        for label_id, class_dir in enumerate(classes_dirs):
            class_dir = os.path.join(root_dir, class_dir)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_id)

        self.crop = crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.crop:
            image = image.crop(self.crop)
        label = self.labels[idx]

        return image, label


class DatasetWithTransform(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        # a imagem do dataset original é um objeto PIL Image
        img, label = self.dataset[idx]

        if self.transform:
            # albumentations espera um array numpy
            img = self.transform(image=np.array(img))['image']

        return img, label

    def __len__(self):
        return len(self.dataset)
