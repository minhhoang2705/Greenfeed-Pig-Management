import torch
import torch.optim as optim
import torch.nn as nn
from data_ingestion import create_data_loader
from pig_detector import PigDetector
import time
import os
import optuna

def train_model(trial, data_dir, annotation_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define hyperparameters to be tuned
    model_path = trial.suggest_categorical('model_path', ['yolov5s', 'yolov5m', 'yolov5l'])
    num_epochs = trial.suggest_int('num_epochs', 5, 15)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    dataloader = create_data_loader(data_dir, annotation_file, batch_size=batch_size)
    # Load pre-trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', model_path, pretrained=True)
    model.to(device)
    
    # For transfer learning, we'll train the entire model but with a lower learning rate
    for param in model.parameters():
        param.requires_grad = True

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
            # Train in autocast
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                
                # YOLOv5 model returns a list of predictions
                if isinstance(outputs, (list, tuple)):
                    # Get the first prediction tensor which contains detection results
                    pred = outputs[0] if len(outputs) > 0 else outputs
                else:
                    pred = outputs
                
                # Convert predictions to the same format as ground truth boxes
                # YOLOv5 outputs are in format [batch_size, num_predictions, 85] where 85 = 4(box) + 1(conf) + 80(class)
                # We only need the box coordinates
                if isinstance(pred, torch.Tensor) and len(pred.shape) == 3:
                    # Extract bounding box coordinates
                    pred_boxes = pred[..., :4]  # Get first 4 values (box coordinates)
                    
                    # Ensure boxes tensor is properly shaped and on the correct device
                    if len(boxes.shape) == 2:
                        boxes = boxes.unsqueeze(0)  # Add batch dimension if needed
                    
                    # Calculate loss only on box coordinates
                    loss = criterion(pred_boxes, boxes.float())
                else:
                    # If predictions are not in expected format, skip this batch
                    continue
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
    return epoch_loss

def objective(trial):
    data_dir = '/home/minhtranh/works/pig_detection/data'
    annotation_file = '/home/minhtranh/works/pig_detection/data/annotations.xml'
    loss = train_model(trial, data_dir, annotation_file)
    return loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_trial)
