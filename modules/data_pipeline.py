import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from PIL import Image
import albumentations as A
from typing import Dict, List, Tuple, Optional
import shutil
from pathlib import Path

class DataPipeline:
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Initialize the data pipeline for preprocessing, augmentation, and splitting.
        
        Args:
            data_dir: Directory containing the original images
            annotation_file: Path to the annotation file
            output_dir: Directory to save processed data
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        # Create output directories
        self.splits = ['train', 'val', 'test']
        for split in self.splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        with open(self.annotation_file) as f:
            self.annotations = json.load(f)
            
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

    def analyze_data_distribution(self) -> Dict:
        """
        Analyze the distribution of bounding boxes and images.
        """
        stats = {
            'total_images': len(self.annotations),
            'boxes_per_image': [],
            'box_sizes': [],
            'box_aspects': []
        }
        
        for img_name, annots in self.annotations.items():
            n_boxes = len(annots)
            stats['boxes_per_image'].append(n_boxes)
            
            for ann in annots:
                bbox = ann['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                stats['box_sizes'].append(width * height)
                stats['box_aspects'].append(width / height if height != 0 else 0)
        
        return stats

    def create_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Create train/val/test splits ensuring balanced distribution.
        Returns lists of image names for each split.
        """
        # Get list of all images
        all_images = list(self.annotations.keys())
        
        # First split into train and temp (val + test)
        train_imgs, temp_imgs = train_test_split(
            all_images,
            train_size=self.train_ratio,
            random_state=self.seed
        )
        
        # Then split temp into val and test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=val_ratio_adjusted,
            random_state=self.seed
        )
        
        return train_imgs, val_imgs, test_imgs

    def verify_splits(self, train_imgs: List[str], val_imgs: List[str], test_imgs: List[str]) -> bool:
        """
        Verify that the splits are correct and balanced.
        """
        # Check for data leakage
        train_set = set(train_imgs)
        val_set = set(val_imgs)
        test_set = set(test_imgs)
        
        assert len(train_set.intersection(val_set)) == 0, "Train-Val overlap detected"
        assert len(train_set.intersection(test_set)) == 0, "Train-Test overlap detected"
        assert len(val_set.intersection(test_set)) == 0, "Val-Test overlap detected"
        
        # Verify split ratios
        total = len(train_imgs) + len(val_imgs) + len(test_imgs)
        actual_ratios = {
            'train': len(train_imgs) / total,
            'val': len(val_imgs) / total,
            'test': len(test_imgs) / total
        }
        
        # Check distribution of boxes per image in each split
        train_boxes = [len(self.annotations[img]) for img in train_imgs]
        val_boxes = [len(self.annotations[img]) for img in val_imgs]
        test_boxes = [len(self.annotations[img]) for img in test_imgs]
        
        # Calculate mean boxes per image for each split
        mean_boxes = {
            'train': np.mean(train_boxes),
            'val': np.mean(val_boxes),
            'test': np.mean(test_boxes)
        }
        
        return {
            'split_ratios': actual_ratios,
            'mean_boxes_per_image': mean_boxes,
            'no_leakage': True
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to an image.
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image

    def get_augmentation(self) -> A.Compose:
        """
        Define augmentation pipeline using albumentations.
        Only applied to training data.
        """
        return A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def process_and_save_splits(self):
        """
        Process images and save them to their respective split directories.
        """
        # Create splits
        train_imgs, val_imgs, test_imgs = self.create_splits()
        
        # Verify splits
        verification = self.verify_splits(train_imgs, val_imgs, test_imgs)
        print("Split verification results:", verification)
        
        # Get augmentation pipeline
        aug = self.get_augmentation()
        
        # Process each split
        split_data = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, images in split_data.items():
            split_annotations = {}
            
            for img_name in images:
                # Load and preprocess image
                img_path = self.data_dir / img_name
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get annotations
                boxes = []
                labels = []
                for ann in self.annotations[img_name]:
                    boxes.append(ann['bbox'])
                    labels.append(0)  # Assuming single class (pig)
                
                # Apply augmentation for training split
                if split_name == 'train':
                    augmented = aug(image=image, bboxes=boxes, labels=labels)
                    image = augmented['image']
                    boxes = augmented['bboxes']
                
                # Preprocess image
                image = self.preprocess_image(image)
                
                # Save processed image
                output_path = self.output_dir / split_name / 'images' / img_name
                cv2.imwrite(str(output_path), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # Update annotations
                split_annotations[img_name] = [
                    {'bbox': box.tolist() if isinstance(box, np.ndarray) else box}
                    for box in boxes
                ]
            
            # Save split annotations
            with open(self.output_dir / split_name / 'annotations.json', 'w') as f:
                json.dump(split_annotations, f, indent=2)

    def generate_data_report(self) -> Dict:
        """
        Generate a comprehensive report about the processed dataset.
        """
        report = {}
        
        for split in self.splits:
            split_dir = self.output_dir / split
            split_ann_file = split_dir / 'annotations.json'
            
            with open(split_ann_file) as f:
                annotations = json.load(f)
            
            report[split] = {
                'num_images': len(annotations),
                'num_annotations': sum(len(anns) for anns in annotations.values()),
                'images_with_boxes': sum(1 for anns in annotations.values() if len(anns) > 0),
                'mean_boxes_per_image': np.mean([len(anns) for anns in annotations.values()])
            }
        
        return report

if __name__ == '__main__':
    # Example usage
    pipeline = DataPipeline(
        data_dir='data',
        annotation_file='annotations.json',
        output_dir='processed_data'
    )
    
    # Analyze original data distribution
    stats = pipeline.analyze_data_distribution()
    print("Original data statistics:", stats)
    
    # Process and save splits
    pipeline.process_and_save_splits()
    
    # Generate report
    report = pipeline.generate_data_report()
    print("Processed data report:", report)
