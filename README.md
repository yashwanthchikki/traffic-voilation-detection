# Videtect

# ViDetect

ViDetect is an AI-powered vehicle detection and number plate recognition system using YOLOv3. The system processes video input to detect vehicles and extract license plate information in real-time.

## Features

- Vehicle detection using YOLOv3
- License plate recognition using Tesseract OCR
- Video frame extraction and processing
- Real-time object detection with bounding boxes
- Support for multiple vehicle types (cars, trucks, motorcycles, etc)

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Tesseract OCR
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ViDetect.git
cd ViDetect
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download YOLOv3 weights:
```bash
# Download from official source or your hosted location
# Place in project root directory
```

## Usage

1. Place your input video in the project directory

2. Run the main script:
```bash
python src/main.py
```

3. Results will be saved in the `output` directory

## Project Structure

```
ViDetect/
├── src/                  # Source code
│   ├── model/           # YOLOv3 model implementation
│   ├── utils/           # Utility functions
│   └── main.py          # Main application
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## License

Owned by Kishor, Yashwanth, Neeraj, Aryann


## Contact

8618510937
