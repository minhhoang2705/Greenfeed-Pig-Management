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
        self.image_dir = os.path.join(data_dir, 'images', 'val2017')  # COCO validation set
        self.annotation_file = annotation_file
        self.transform = transform
        
        # Load COCO annotations
        if annotation_file:
            with open(annotation_file, 'r') as f:
                self.coco_data = json.load(f)
            
            # Create image id to filename mapping
            self.image_id_to_filename = {
                img['id']: img['file_name'] for img in self.coco_data['images']
            }
            
            # Create image id to annotations mapping
            self.image_id_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_id_to_anns:
                    self.image_id_to_anns[img_id] = []
                self.image_id_to_anns[img_id].append(ann)
            
            # Get list of image IDs that have annotations
            self.image_ids = list(self.image_id_to_anns.keys())
        else:
            self.image_ids = []
            self.coco_data = None

    def _load_image(self, image_id):
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        return Image.open(image_path).convert('RGB'), filename

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, image_name = self._load_image(image_id)
        
        boxes = []
        labels = []
        if image_id in self.image_id_to_anns:
            for ann in self.image_id_to_anns[image_id]:
                # COCO bbox format is [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(0)  # Assuming all objects are pigs (class 0)

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
