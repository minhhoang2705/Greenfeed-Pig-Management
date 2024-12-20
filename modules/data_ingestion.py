import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PigDataset(Dataset):
    def __init__(self, data_dir, annotation_file=None, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.annotation_file = annotation_file
        self.annotations = self._load_annotations() if annotation_file else {}
        self.transform = transform

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        boxes = []
        labels = []
        if image_name in self.annotations:
            for annotation in self.annotations[image_name]:
                boxes.append(annotation['bbox'])
                labels.append(0) # Assuming pig is class 0

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels, image_name

def create_data_loader(data_dir, annotation_file=None, batch_size=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PigDataset(data_dir, annotation_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    # Example usage
    data_dir = 'data' # Replace with your data directory
    annotation_file = 'annotations.json' # Replace with your annotation file
    dataloader = create_data_loader(data_dir, annotation_file)

    for images, boxes, labels, image_names in dataloader:
        print("Image batch shape:", images.shape)
        print("Boxes:", boxes)
        print("Labels:", labels)
        print("Image names:", image_names)
        break
