import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
import sys

# ===== CONFIGURATION PARAMETERS =====
# Adjust these values to fine-tune detection performance:

RIDER_CONF = 0.6          # Confidence threshold for rider detection
HELMET_CONF = 0.7         # Confidence threshold for helmet detection (class 1) - INCREASE to reduce false helmet detections
NO_HELMET_CONF = 0.75      # Confidence threshold for no_helmet detection (class 2) - DECREASE to catch more violations
OVERLAP_THRESHOLD = 0.5   # Overlap threshold for helmet-rider association (0.0-1.0)

# ===== END CONFIGURATION =====

# Initialize model and tracker
model = YOLO("five.pt")
print(f"Model classes: {model.names}")  # Print model classes
tracker = sv.ByteTrack()

# Output directories
output_dir = Path("helmet_violations")
output_dir.mkdir(exist_ok=True)
snapshots_dir = output_dir / "snapshots"
snapshots_dir.mkdir(exist_ok=True)

# Tracked violations to prevent duplicates
violated_riders = set()

def filter_helmet_detections(detections):
    """Filter helmet detections based on class-specific confidence thresholds"""
    if len(detections) == 0 or detections.class_id is None:
        return detections
    
    # Create mask for filtering based on class-specific confidence
    keep_mask = np.zeros(len(detections), dtype=bool)
    
    for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
        if class_id == 1 and confidence >= HELMET_CONF:  # Helmet
            keep_mask[i] = True
        elif class_id == 2 and confidence >= NO_HELMET_CONF:  # No helmet
            keep_mask[i] = True
    
    # Apply the mask to filter detections
    if np.any(keep_mask):
        return detections[keep_mask]
    else:
        # Return empty detections object
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int)
        )

def is_inside(inner_box, outer_box, overlap_threshold=0.5):
    """Check if inner_box has significant overlap with outer_box"""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Calculate intersection
    x_left = max(ix1, ox1)
    y_top = max(iy1, oy1)
    x_right = min(ix2, ox2)
    y_bottom = min(iy2, oy2)
    
    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return False
        
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    
    # Check if overlap is significant
    return intersection_area / inner_area > overlap_threshold

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
        
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect objects with higher confidence for riders
        results = model(frame, conf=RIDER_CONF, classes=[0])  # Use configurable rider confidence
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        if detections.tracker_id is not None:  # Add this check
            for rider_idx in range(len(detections)):
                rider_box = detections.xyxy[rider_idx]
                rider_id = detections.tracker_id[rider_idx]
                
                # Look for helmet/no-helmet ONLY within this rider box
                helmet_results = model(frame, conf=0.3, classes=[1, 2])  # Use low conf to catch all, then filter
                helmet_detections = sv.Detections.from_ultralytics(helmet_results[0])
                
                # Apply class-specific confidence filtering
                helmet_detections = filter_helmet_detections(helmet_detections)
                
                # Debug: Print helmet detections
                if len(helmet_detections) > 0 and frame_count % 30 == 0:  # Every 30 frames
                    print(f"Frame {frame_count}: Found {len(helmet_detections)} helmet/no-helmet detections")
                    if helmet_detections.class_id is not None and helmet_detections.confidence is not None:
                        for i, (cls, conf) in enumerate(zip(helmet_detections.class_id, helmet_detections.confidence)):
                            print(f"  - Class {cls} (conf: {conf:.2f})")
                
                violation_detected = False
                if helmet_detections.class_id is not None:  # Add this check
                    for helmet_idx in range(len(helmet_detections)):
                        helmet_box = helmet_detections.xyxy[helmet_idx]
                        if is_inside(helmet_box, rider_box, OVERLAP_THRESHOLD):
                            if helmet_detections.class_id[helmet_idx] == 2:
                                violation_detected = True
                
                # Handle violation
                if violation_detected and rider_id not in violated_riders:
                    violated_riders.add(rider_id)
                    
                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"violation_rider_{rider_id}_{timestamp}.jpg"
                    cv2.imwrite(str(snapshots_dir / filename), frame)
                    
                    print(f"🚨 VIOLATION DETECTED: Rider {rider_id} at frame {frame_count}")
        
        # Visualize
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Draw rider boxes
        rider_mask = detections.class_id == 0
        rider_detections = detections[rider_mask]
        
        # Create labels only if tracker_id exists
        labels = []
        if hasattr(rider_detections, 'tracker_id') and rider_detections.tracker_id is not None:
            labels = [f"Rider {tid}" for tid in rider_detections.tracker_id]
        
        # Only annotate if we have detections
        if len(rider_detections) > 0:
            annotated_frame = box_annotator.annotate(
                frame.copy(), 
                rider_detections,
                labels=labels
            )
        else:
            annotated_frame = frame.copy()
            
        # Draw helmet detections with different colors, but ONLY if they're inside a rider box
        helmet_results = model(frame, conf=0.3, classes=[1, 2])
        helmet_detections = sv.Detections.from_ultralytics(helmet_results[0])
        
        # Apply class-specific confidence filtering
        helmet_detections = filter_helmet_detections(helmet_detections)
        
        # Only process helmet detections if we have rider detections to compare with
        if len(rider_detections) > 0 and isinstance(rider_detections, sv.Detections) and len(helmet_detections) > 0 and helmet_detections.class_id is not None:
            # For each helmet detection, check if it's inside any rider box
            for i, (box, class_id) in enumerate(zip(helmet_detections.xyxy, helmet_detections.class_id)):
                # Check if this helmet/no-helmet is inside any rider box
                is_inside_any_rider = False
                for rider_box in rider_detections.xyxy:
                    if is_inside(box, rider_box, OVERLAP_THRESHOLD):
                        is_inside_any_rider = True
                        break
                
                # Only draw if inside a rider box
                if is_inside_any_rider:
                    # Set color based on class
                    if class_id == 1:  # Helmet
                        color = (0, 255, 0)  # Green
                        label = "Helmet"
                    else:  # No-helmet
                        color = (0, 0, 255)  # Red
                        label = "No Helmet"
                    
                    # Draw the box and label
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show frame
        cv2.imshow("Helmet Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default video path
    video_path = "video_1.mp4"
    
    # Check if a video path is provided as a command-line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
    print(f"Processing video: {video_path}")
    process_video(video_path)
    print(f"Processing complete. Snapshots saved to {snapshots_dir}")