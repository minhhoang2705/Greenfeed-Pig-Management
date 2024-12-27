import os
import json
import requests
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm

class COCODatasetPreparation:
    def __init__(self):
        self.dataset_dir = Path("datasets/coco")
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        # COCO URLs
        self.urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        # Create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_path: Path):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def extract_zip(self, zip_path: Path, extract_path: Path):
        """Extract a zip file with progress bar"""
        with ZipFile(zip_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc=f"Extracting {zip_path.name}"):
                zip_ref.extract(file, extract_path)

    def download_and_prepare_dataset(self):
        """Download and prepare COCO dataset"""
        print("Downloading and preparing COCO dataset...")
        
        # Download and extract datasets
        for name, url in self.urls.items():
            zip_path = self.dataset_dir / f"{name}.zip"
            
            # Download if not exists
            if not zip_path.exists():
                print(f"\nDownloading {name}...")
                self.download_file(url, zip_path)
            
            # Extract if needed
            print(f"\nExtracting {name}...")
            self.extract_zip(zip_path, self.dataset_dir)
            
            # Clean up zip file
            zip_path.unlink()

        print("\nFiltering annotations for pigs...")
        # Process annotations to filter pig-related data
        self.filter_pig_annotations()
        
        print("\nDataset preparation completed!")

    def convert_bbox_coco_to_yolo(self, bbox, img_width, img_height):
        """Convert COCO bbox to YOLO format"""
        # COCO: [x_min, y_min, width, height]
        # YOLO: [x_center, y_center, width, height] (normalized)
        x_min, y_min, width, height = bbox
        
        # Convert to YOLO format (normalized)
        x_center = (x_min + width/2) / img_width
        y_center = (y_min + height/2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        return [x_center, y_center, norm_width, norm_height]

    def filter_pig_annotations(self):
        """Filter annotations and convert to YOLO format"""
        for split in ['train', 'val']:
            ann_file = self.annotations_dir / f'instances_{split}2017.json'
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Find pig category ID
            pig_id = None
            for cat in data['categories']:
                if cat['name'] == 'pig':
                    pig_id = cat['id']
                    break
            
            if pig_id is None:
                print(f"Warning: No pig category found in {split} annotations")
                continue
            
            # Filter annotations
            pig_anns = [ann for ann in data['annotations'] if ann['category_id'] == pig_id]
            pig_img_ids = set(ann['image_id'] for ann in pig_anns)
            pig_images = [img for img in data['images'] if img['id'] in pig_img_ids]
            
            # Create image id to info mapping
            img_info = {img['id']: img for img in pig_images}
            
            # Group annotations by image
            img_to_anns = {}
            for ann in pig_anns:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)
            
            # Create YOLO format annotations
            print(f"\nConverting {split} annotations to YOLO format...")
            for img_id, anns in tqdm(img_to_anns.items()):
                img = img_info[img_id]
                img_width, img_height = img['width'], img['height']
                
                # Create YOLO format annotations
                yolo_anns = []
                for ann in anns:
                    bbox = self.convert_bbox_coco_to_yolo(ann['bbox'], img_width, img_height)
                    # Class id is 0 since we only have one class (pig)
                    yolo_anns.append(f"0 {' '.join(map(str, bbox))}")
                
                # Save YOLO format annotations
                label_path = self.dataset_dir / f"{split}2017" / f"{img['file_name'].replace('.jpg', '.txt')}"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_anns))
            
            print(f"Processed {len(pig_anns)} pig annotations in {len(pig_images)} images for {split}")

if __name__ == "__main__":
    dataset_prep = COCODatasetPreparation()
    dataset_prep.download_and_prepare_dataset()
