# Helmet Detection System

A comprehensive helmet violation detection system using YOLOv8 for real-time video processing. This system detects motorcycle riders and identifies helmet violations with advanced tracking capabilities.

## 🚀 Features

- **Real-time Video Processing**: Process videos with helmet violation detection
- **Advanced Tracking**: ByteTrack/DeepSORT tracking for consistent vehicle identification
- **Class-Specific Confidence**: Adjustable confidence thresholds for different classes
- **Violation Scoring**: Intelligent violation detection with spatial analysis
- **Comprehensive Logging**: Detailed logs and violation snapshots
- **Multiple Model Support**: Works with different trained models (best.pt, five.pt, etc.)

## 📁 Project Structure

```
helmet-dataset/
├── training.py          # YOLOv8 training script
├── detector.py          # Helmet violation detection system
├── data.yaml           # Dataset configuration
├── best.pt             # Trained model weights
├── five.pt             # Alternative model weights
├── video_1.mp4         # Test videos
├── video_2.mp4
├── video_3.mp4
├── requirements.txt    # Python dependencies
└── helmet_violations_output/  # Detection results
    ├── snapshots/      # Violation images
    └── reports/        # JSON reports
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
   python detector.py --help
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

Use `detector.py` for helmet violation detection:

```bash
# Basic usage
python detector.py --video video_1.mp4 --model best.pt

# With custom confidence thresholds
python detector.py --video video_2.mp4 --model five.pt --confidence 0.3

# With custom violation threshold
python detector.py --video video_3.mp4 --model best.pt --violation_threshold 0.4

# Using DeepSORT tracker
python detector.py --video video_1.mp4 --model best.pt --tracker deepsort
```

## ⚙️ Model Configuration

### Available Models

- **best.pt**: High-accuracy model (84MB)
- **five.pt**: Optimized model (6MB)

### Confidence Threshold Adjustment

**⚠️ Important**: You need to adjust confidence thresholds for optimal performance with different models.

#### For best.pt model:
```python
# Recommended settings in detector.py
self.class_confidences = {
    0: 0.2,  # Bike_Rider - low confidence to catch all riders
    1: 0.3,  # Helmet - moderate confidence
    2: 0.3   # No_Helmet - moderate confidence
}
```

#### For five.pt model:
```python
# Recommended settings in detector.py
self.class_confidences = {
    0: 0.1,  # Bike_Rider - very low confidence
    1: 0.2,  # Helmet - lower confidence
    2: 0.2   # No_Helmet - lower confidence
}
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Input video path | `video_1.mp4` |
| `--model` | Model weights path | `best.pt` |
| `--confidence` | Global confidence threshold | `0.1` |
| `--violation_threshold` | Violation detection threshold | `0.3` |
| `--tracker` | Tracker type (bytetrack/deepsort) | `bytetrack` |
| `--output` | Output video path (optional) | None |

## 📊 Output

The system generates comprehensive outputs:

### Violation Snapshots
- High-quality images of detected violations
- Saved in `helmet_violations_output/snapshots/`
- Filename format: `violation_track_{ID}_{timestamp}_score_{score}.jpg`

### Reports
- JSON summary reports with violation details
- Saved in `helmet_violations_output/reports/`
- Includes frame numbers, scores, and detection counts

### Logs
- Detailed processing logs in `helmet_detection.log`
- Real-time progress updates and violation alerts

## 🎮 Class Definitions

The system detects three main classes:

- **Bike_Rider (0)**: Motorcycle riders
- **Helmet (1)**: Riders wearing helmets
- **No_Helmet (2)**: Riders without helmets

## 🔧 Performance Optimization

### For best.pt:
- **Speed**: ~250ms per frame
- **Accuracy**: High detection accuracy
- **Memory**: Higher memory usage

### For five.pt:
- **Speed**: ~35ms per frame
- **Accuracy**: Good balance of speed and accuracy
- **Memory**: Lower memory usage

## 📝 Example Output

```
2025-07-02 05:26:47,793 - INFO - Processing complete!
2025-07-02 05:26:47,793 - INFO - Total violations detected: 9
2025-07-02 05:26:47,794 - WARNING - 🚨 9 helmet violations detected and saved!
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