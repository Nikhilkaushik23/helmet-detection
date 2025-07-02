# Helmet Detection System

A comprehensive helmet violation detection system using YOLOv8 for real-time video processing. This system detects motorcycle riders and identifies helmet violations with advanced tracking capabilities.

## 🚀 Features

- **Real-time Video Processing**: Process videos with helmet violation detection
- **Advanced Tracking**: ByteTrack tracking for consistent rider identification
- **Class-Specific Confidence**: Adjustable confidence thresholds for different classes
- **Violation Scoring**: Intelligent violation detection with spatial analysis
- **Comprehensive Logging**: Violation snapshots with timestamps
- **Multiple Model Support**: Works with different trained models (best.pt, five.pt, etc.)

## 📁 Project Structure

```
helmet-dataset/
├── training.py          # YOLOv8 training script
├── detect.py            # Helmet detection script
├── data.yaml            # Dataset configuration
├── best.pt              # Trained model weights
├── five.pt              # Alternative model weights
├── video_1.mp4          # Test videos
├── video_2.mp4
├── video_3.mp4
├── requirements.txt     # Python dependencies
└── helmet_violations/   # Detection results
    └── snapshots/       # Violation images
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd helmet-dataset
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python detect.py --help
   ```

## 🎯 Usage

### Training a New Model

Use `training.py` to train a custom YOLOv8 model:

```bash
python training.py
```

**Training Configuration:**
- Optimized for RTX 3060 GPU (12GB VRAM)
- Automatic batch size optimization
- Early stopping and checkpointing
- Comprehensive logging

### Running Detection

```bash
# Basic usage
python detect.py video_1.mp4

# With custom video path
python detect.py feed1.mp4
```

The `detect.py` script provides:
- Efficient rider tracking using ByteTrack
- Class-specific confidence thresholds
- Spatial filtering to only detect helmet violations within rider boxes
- Automatic snapshot saving of violations

## ⚙️ Model Configuration

### Available Models

- **best.pt**: High-accuracy model (84MB)
- **five.pt**: Optimized model (6MB)

### Confidence Threshold Adjustment

Edit the configuration parameters at the top of the file:

```python
# ===== CONFIGURATION PARAMETERS =====
RIDER_CONF = 0.6          # Confidence threshold for rider detection
HELMET_CONF = 0.7         # Confidence threshold for helmet detection (class 1)
NO_HELMET_CONF = 0.75     # Confidence threshold for no_helmet detection (class 2)
OVERLAP_THRESHOLD = 0.5   # Overlap threshold for helmet-rider association
```

- Increase `HELMET_CONF` to reduce false helmet detections
- Decrease `NO_HELMET_CONF` to catch more potential violations
- Adjust `OVERLAP_THRESHOLD` to control how much overlap is required between helmet/no-helmet and rider boxes

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Input video path | `video_1.mp4` |

## 📊 Output

- Violation snapshots saved to `helmet_violations/snapshots/`
- Real-time visualization with color-coded bounding boxes:
  - Green: Helmet detected
  - Red: No helmet detected
  - Blue: Rider detected
- Console output with violation alerts

## 🎮 Class Definitions

The system detects three main classes:

- **Rider (0)**: Motorcycle riders
- **With Helmet (1)**: Riders wearing helmets
- **Without Helmet (2)**: Riders without helmets

## 🔧 Performance Optimization

- **Speed**: ~35ms per frame with five.pt model
- **Memory**: Efficient memory usage
- **Accuracy**: High precision with spatial filtering

## 📝 Example Output

```
🚨 VIOLATION DETECTED: Rider 6 at frame 333
Processing complete. Snapshots saved to helmet_violations/snapshots
```

## 🐛 Troubleshooting

### Common Issues

1. **Low detection rate**: Lower confidence thresholds
2. **High false positives**: Increase confidence thresholds
3. **Slow processing**: Use five.pt model or reduce video resolution
4. **Memory issues**: Use smaller batch sizes or five.pt model

### Model-Specific Tips

- **best.pt**: Use for high-accuracy applications, adjust confidence to 0.2-0.4
- **five.pt**: Use for real-time applications, adjust confidence to 0.1-0.3

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 