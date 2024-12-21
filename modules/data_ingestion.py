import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class PigDataset(Dataset):
    def __init__(self, data_dir, annotation_file=None, transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.annotation_file = annotation_file
        self.annotations = self._load_annotations() if annotation_file else {}
        self.transform = transform

    def _load_annotations(self):
        annotations = {}
        if self.annotation_file:
            tree = ET.parse(self.annotation_file)
            root = tree.getroot()
            for image in root.findall('image'):
                file_element = image.find('file')
                if file_element is not None:
                    image_name = file_element.text
                    image_annotations = []
                    if image_name in self.image_files:
                        for box in image.findall('box'):
                            xmin = int(box.find('xmin').text)
                            ymin = int(box.find('ymin').text)
                            xmax = int(box.find('xmax').text)
                            ymax = int(box.find('ymax').text)
                            bbox = [xmin, ymin, xmax, ymax]
                            image_annotations.append({'bbox': bbox})
                        annotations[image_name] = image_annotations
        return annotations

    def __len__(self):
        print(f"Number of images loaded: {len(self.image_files)}")
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
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
