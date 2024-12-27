import torch
import os
from ultralytics import YOLO


def train_model(data_yaml='/home/minhtranh/works/pig_detection/combined_dataset/data.yaml'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize YOLOv8 model
    model = YOLO('yolov8s.pt')  # Using small model for YOLOv8
    
    # Training hyperparameters optimized for YOLOv11
    epochs = 100  # Increased epochs for better convergence
    batch_size = 32  # Adjusted batch size
    learning_rate = 1e-3  # Initial learning rate
    
    print("Starting training with YOLOv8...")
    print(f"Training configuration:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Device: {device}")
    
    # Train the model using YOLO's built-in training method
    try:
        results = model.train(
            data=data_yaml,  # Using the provided data.yaml
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            device=device,
            imgsz=640,
            patience=50,  # Early stopping patience
            save_period=10,  # Save checkpoint every 10 epochs
            cache=True,  # Cache images for faster training
            amp=True,  # Use mixed precision training
            optimizer='AdamW'  # Using AdamW optimizer
        )
        print("Training complete!")
        
        # Validate the model
        print("Running validation...")
        model.val()
        
        return results
    except Exception as e:
        print(f"Training error: {str(e)}")
        return None


if __name__ == '__main__':
    train_model()
