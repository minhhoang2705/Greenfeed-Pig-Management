# Pig Detection System

This project provides a pig detection system using YOLO-based object detection and tracking. It consists of a FastAPI backend with Celery for asynchronous video processing and a Streamlit-based web interface.

## Features
- Real-time pig detection in images and videos
- Asynchronous video processing with Celery
- REST API for easy integration
- Streamlit-based web interface

## Setup

### Prerequisites
- Python 3.8+ (recommended to use pyenv for version management)
- Redis server (version 6.0+)
- CUDA-enabled GPU (optional but recommended for faster processing)
- NVIDIA drivers and CUDA toolkit (if using GPU)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pig-detection.git
   cd pig-detection
   ```

2. **Set up Python environment**:
   ```bash
   # Install pyenv (if not already installed)
   curl https://pyenv.run | bash

   # Install specific Python version
   pyenv install 3.8.12

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install system dependencies**:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
   libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
   liblzma-dev python-openssl git redis-server

   # For macOS
   brew install openssl readline sqlite3 xz zlib redis
   ```

4. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

   # For GPU support (if available)
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
   ```

5. **Configure Redis**:
   ```bash
   # Start Redis server
   redis-server --daemonize yes

   # Verify Redis is running
   redis-cli ping
   ```

6. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```bash
   REDIS_URL=redis://localhost:6379/0
   MODEL_PATH=weights/best.pt
   ```

7. **Download model weights**:
   Place your trained YOLO model weights in the `weights/` directory

## Running the Application

### Start FastAPI Server
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Start Celery Worker
```bash
celery -A api.app.celery worker --loglevel=info --concurrency=4
```

### Start Streamlit UI
```bash
streamlit run frontend/ui.py
```

## Using the System

### API Endpoints

1. **Image Detection**:
   ```bash
   curl -X POST -F "file=@test_image.jpg" http://localhost:8000/api/v1/detect/image
   ```

2. **Video Processing**:
   ```bash
   curl -X POST -F "file=@test_video.avi" http://localhost:8000/api/v1/detect/video
   ```

3. **Get Result File**:
   ```bash
   # For images
   curl -o result.jpg http://localhost:8000/api/v1/results/result_image_YYYYMMDD_HHMMSS.jpg
   # For videos
   curl -o result.avi http://localhost:8000/api/v1/results/result_video_YYYYMMDD_HHMMSS.avi
   ```

4. **Check API Health**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Web Interface
Access the web interface at http://localhost:8501

## Testing the System

1. **Run unit tests**:
   ```bash
   pytest tests/
   ```

2. **Run integration tests**:
   ```bash
   python -m pytest tests/integration/
   ```

3. **Check system health**:
   ```bash
   curl http://localhost:8000/health
   ```

## Configuration

The system can be configured through the following files:
- `config.yaml`: Main configuration file
- `modules/configs/config.yaml`: Model-specific configurations

## Monitoring Tasks

You can monitor the status of video processing tasks using the `/task/{task_id}` endpoint. The task will go through these states:
1. PENDING: Task is queued
2. STARTED: Processing has begun
3. SUCCESS: Processing completed successfully
4. FAILURE: Processing failed (check status for details)

## Troubleshooting

### Celery Worker Issues
- Ensure Redis server is running
- Verify Celery worker is started with correct application path
- Check task names match between API and worker

### GPU Acceleration
- Install CUDA toolkit if using GPU
- Verify PyTorch is using CUDA:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
