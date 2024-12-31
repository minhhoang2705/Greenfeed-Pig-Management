import torch
import os
from ultralytics import YOLO
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {}

def train_model(config_path='/home/minhtranh/works/pig_detection/modules/configs/config.yaml'):
    """
    Trains a YOLOv8 model based on the provided configuration.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load model
    model_name = config.get('model_name', 'yolov8s.pt')
    if model_name.endswith('.pt'):
      model = YOLO(model_name)
    else:
      model = YOLO(f'{model_name}.pt')

    # Training hyperparameters
    data_yaml = config.get('data_yaml', '/home/minhtranh/works/pig_detection/combined_dataset/data.yaml')
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 1e-3)
    imgsz = config.get('imgsz', 640)
    patience = config.get('patience', 50)
    save_period = config.get('save_period', 10)
    cache = config.get('cache', True)
    amp = config.get('amp', True)
    optimizer = config.get('optimizer', 'AdamW')
    project = config.get('project', None) # Default project name
    name = config.get('name', None)  # Default experiment name

    logging.info("Starting training with YOLOv8...")
    logging.info(f"Training configuration:")
    logging.info(f"- Model: {model_name}")
    logging.info(f"- Data YAML: {data_yaml}")
    logging.info(f"- Epochs: {epochs}")
    logging.info(f"- Batch size: {batch_size}")
    logging.info(f"- Learning rate: {learning_rate}")
    logging.info(f"- Image size: {imgsz}")
    logging.info(f"- Device: {device}")

    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=learning_rate,
            device=device,
            imgsz=imgsz,
            patience=patience,
            save_period=save_period,
            cache=cache,
            amp=amp,
            optimizer=optimizer,
            project=project,
            name=name
        )
        logging.info("Training complete!")

        # Validate the model
        logging.info("Running validation...")
        val_results = model.val(data=data_yaml, device=device, imgsz=imgsz)

        return results, val_results
    except Exception as e:
        logging.error(f"Training error: {e}")
        return None, None

if __name__ == '__main__':
    train_model()