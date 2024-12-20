import torch
import torch.optim as optim
import torch.nn as nn
from data_ingestion import create_data_loader
from pig_detector import PigDetector
import time
import os

def train_model(data_dir, annotation_file, model_path='yolov5s', num_epochs=10, batch_size=4, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = create_data_loader(data_dir, annotation_file, batch_size=batch_size)
    model = torch.hub.load('ultralytics/yolov5', model_path, pretrained=True)
    model.to(device)
    
    # Freeze all layers except the last one for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    model.model[-1].requires_grad_(True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Using MSE loss for bounding box regression

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for images, boxes, labels, _ in dataloader:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Assuming the output is a tuple, and the bounding box predictions are in the first element
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Calculate loss
            loss = criterion(predictions, boxes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        end_time = time.time()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/trained_pig_detector.pth')
    print("Training complete. Model saved to models/trained_pig_detector.pth")

if __name__ == '__main__':
    # Example usage
    data_dir = 'data' # Replace with your data directory
    annotation_file = 'annotations.json' # Replace with your annotation file
    train_model(data_dir, annotation_file)
