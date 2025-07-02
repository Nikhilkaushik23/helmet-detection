import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict, deque
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('helmet_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HelmetViolationDetector:
    def __init__(self, model_path="best.pt", confidence_threshold=0.1, tracker_type="bytetrack"):
        """
        Initialize the helmet violation detector.
        
        Args:
            model_path (str): Path to the YOLO model weights
            confidence_threshold (float): Minimum confidence for detections
            tracker_type (str): Type of tracker to use ('bytetrack' or 'deepsort')
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class mapping for 3-class model: 0=Bike_Rider, 1=Helmet, 2=No_Helmet
        self.class_names = ['Bike_Rider', 'Helmet', 'No_Helmet']
        
        # Class-specific confidence thresholds
        self.class_confidences = {
            0: 0.2,  # Bike_Rider - very low confidence to catch all riders
            1: 0.3,   # Helmet - higher confidence for helmet detection
            2: 0.3    # No_Helmet - higher confidence for no-helmet detection
        }
        
        # Initialize tracker
        if tracker_type.lower() == "deepsort":
            self.tracker = sv.DeepSORT()
        else:
            self.tracker = sv.ByteTrack()
        
        # Violation tracking
        self.violation_history = defaultdict(list)  # track_id -> list of violation scores
        self.flagged_vehicles = set()  # Set of track_ids that have been flagged
        self.best_violations = {}  # track_id -> best violation data
        
        # Detection history for smoothing
        self.detection_history = defaultdict(lambda: deque(maxlen=10))
        
        # Output directories - fixed directory structure
        self.setup_output_dirs()
        
    def setup_output_dirs(self):
        """Create fixed output directories for saving results."""
        # Use fixed directory names instead of timestamped ones
        self.output_dir = Path("helmet_violations_output")
        self.snapshots_dir = self.output_dir / "snapshots"
        self.reports_dir = self.output_dir / "reports"
        
        self.output_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directories created: {self.output_dir}")
        
    def apply_class_specific_confidence(self, detections):
        """
        Apply class-specific confidence thresholds to filter detections.
        
        Args:
            detections: Supervision detections object
            
        Returns:
            Filtered detections object
        """
        if len(detections) == 0 or detections.class_id is None:
            return detections
            
        # Create mask for each class based on their specific confidence threshold
        class_mask = np.zeros(len(detections), dtype=bool)
        
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            if class_id in self.class_confidences:
                threshold = self.class_confidences[class_id]
                if confidence >= threshold:
                    class_mask[i] = True
            else:
                # For unknown classes, use default threshold
                if confidence >= self.confidence_threshold:
                    class_mask[i] = True
        
        # Apply the mask to filter detections
        if np.any(class_mask):
            filtered_detections = detections[class_mask]
            logger.debug(f"Filtered detections: {len(detections)} -> {len(filtered_detections)}")
            return filtered_detections
        else:
            # Return original detections if no filtering needed
            return detections
    
    def calculate_violation_score(self, detections, boxes):
        """
        Calculate violation score based on STRICT RIDER-ONLY detection strategy.
        Only considers helmet detections that are spatially contained within rider boxes.
        Ignores standalone helmet detections (pedestrians) completely.
        
        Args:
            detections: Supervision detections object
            boxes: Bounding boxes
            
        Returns:
            dict: track_id -> violation_score
        """
        violation_scores = {}
        
        # First, identify all rider detections across all tracks
        all_rider_boxes = []
        all_rider_track_ids = []
        
        for track_id in np.unique(detections.tracker_id):
            if track_id == -1:  # Skip untracked detections
                continue
                
            track_mask = detections.tracker_id == track_id
            track_classes = detections.class_id[track_mask]
            track_boxes = detections.xyxy[track_mask]
            
            # Find rider detections for this track
            rider_mask = track_classes == 0  # Bike_Rider (class 0)
            if np.any(rider_mask):
                rider_boxes_for_track = track_boxes[rider_mask]
                for rider_box in rider_boxes_for_track:
                    all_rider_boxes.append(rider_box)
                    all_rider_track_ids.append(track_id)
        
        # Now process each track, but only consider helmets within ANY rider box
        for track_id in np.unique(detections.tracker_id):
            if track_id == -1:  # Skip untracked detections
                continue
                
            # Get all detections for this track_id
            track_mask = detections.tracker_id == track_id
            track_classes = detections.class_id[track_mask]
            track_confidences = detections.confidence[track_mask]
            track_boxes = detections.xyxy[track_mask]
            
            # Count riders in this track
            riders = np.sum(track_classes == 0)  # Bike_Rider (class 0)
            
            # STRICT RIDER-ONLY LOGIC: Only process if rider is detected in this track
            if riders == 0:
                # No rider detected - skip this track entirely
                violation_scores[track_id] = 0.0
                continue
                
            # Get rider bounding boxes for this specific track
            rider_mask = track_classes == 0  # Bike_Rider (class 0)
            rider_boxes = track_boxes[rider_mask]
            rider_confidences = track_confidences[rider_mask]
            
            # Check helmet detections ONLY within THIS track's rider boxes
            helmets_in_rider_area = 0
            violations_in_rider_area = 0
            
            for i, detection_class in enumerate(track_classes):
                if detection_class in [1, 2]:  # helmet classes (Helmet=1, No_Helmet=2)
                    helmet_box = track_boxes[i]
                    helmet_confidence = track_confidences[i]
                    
                    # Check if helmet detection is contained within THIS track's rider boxes
                    helmet_associated = False
                    for rider_box in rider_boxes:
                        if self._is_helmet_in_rider_box(helmet_box, rider_box):
                            helmet_associated = True
                            if detection_class == 1:  # Helmet
                                helmets_in_rider_area += 1
                            elif detection_class == 2:  # No_Helmet
                                violations_in_rider_area += 1
                            break
                    
                    # If helmet is not associated with any rider in this track, ignore it
                    # This effectively filters out pedestrian helmet detections
            
            # Calculate violation score based on helmet status within rider area
            violation_score = 0.0
            
            if violations_in_rider_area > 0:
                # Direct violation: 'No_Helmet' class detected within rider box
                violation_score = 1.0
            elif helmets_in_rider_area == 0 and riders > 0:
                # No helmet detected for rider - potential violation
                violation_score = 0.8  # Increased from 0.7 for better detection
            else:
                # Rider with helmet detected - compliant
                violation_score = 0.0
                
            # Weight by rider confidence
            if len(rider_confidences) > 0:
                avg_rider_confidence = np.mean(rider_confidences)
                violation_scores[track_id] = violation_score * avg_rider_confidence
            else:
                violation_scores[track_id] = 0.0
            
        return violation_scores
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """
        Check if two bounding boxes overlap significantly.
        
        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2]
            threshold: Minimum overlap ratio to consider as overlap
            
        Returns:
            bool: True if boxes overlap significantly
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
            
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate overlap ratio (intersection over union)
        union_area = area1 + area2 - intersection_area
        overlap_ratio = intersection_area / union_area if union_area > 0 else 0
        
        return overlap_ratio >= threshold
    
    def _is_helmet_in_rider_box(self, helmet_box, rider_box, containment_threshold=0.6):
        """
        Check if a helmet detection is properly contained within a rider bounding box.
        Uses strict spatial containment to filter out pedestrian helmet detections.
        
        Args:
            helmet_box: Helmet bounding box [x1, y1, x2, y2]
            rider_box: Rider bounding box [x1, y1, x2, y2]
            containment_threshold: Minimum ratio of helmet box that must be inside rider box
            
        Returns:
            bool: True if helmet is contained within rider box
        """
        h_x1, h_y1, h_x2, h_y2 = helmet_box
        r_x1, r_y1, r_x2, r_y2 = rider_box
        
        # Expand rider box slightly to account for detection imprecision
        margin = 10  # pixels
        r_x1 -= margin
        r_y1 -= margin  
        r_x2 += margin
        r_y2 += margin
        
        # Calculate intersection area
        intersection_x1 = max(h_x1, r_x1)
        intersection_y1 = max(h_y1, r_y1)
        intersection_x2 = min(h_x2, r_x2)
        intersection_y2 = min(h_y2, r_y2)
        
        # No intersection
        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return False
        
        # Calculate areas
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        helmet_area = (h_x2 - h_x1) * (h_y2 - h_y1)
        
        # Check if sufficient portion of helmet is contained within rider box
        if helmet_area <= 0:
            return False
            
        containment_ratio = intersection_area / helmet_area
        
        # Also check if helmet center is within rider box (additional constraint)
        helmet_center_x = (h_x1 + h_x2) / 2
        helmet_center_y = (h_y1 + h_y2) / 2
        center_in_rider = (r_x1 <= helmet_center_x <= r_x2) and (r_y1 <= helmet_center_y <= r_y2)
        
        # Helmet is associated with rider if:
        # 1. Sufficient containment ratio AND 2. Helmet center is within rider box
        return containment_ratio >= containment_threshold and center_in_rider
    
    def smooth_violations(self, track_id, current_score):
        """
        Smooth violation scores over time to reduce false positives.
        
        Args:
            track_id: Vehicle tracking ID
            current_score: Current violation score
            
        Returns:
            float: Smoothed violation score
        """
        self.detection_history[track_id].append(current_score)
        
        # Calculate moving average
        history = list(self.detection_history[track_id])
        if len(history) >= 3:  # Need at least 3 frames for reliable detection
            smoothed_score = np.mean(history[-5:])  # Average of last 5 frames
            return smoothed_score
        
        return 0.0  # Not enough history
    
    def update_best_violation(self, track_id, violation_score, frame, detections, frame_number):
        """
        Update the best violation snapshot for a track_id.
        Focus on capturing the rider bounding box when violation is detected.
        
        Args:
            track_id: Vehicle tracking ID
            violation_score: Current violation score
            frame: Current frame
            detections: Current detections
            frame_number: Current frame number
        """
        if track_id not in self.best_violations or violation_score > self.best_violations[track_id]['score']:
            # Get bounding box for this track_id
            track_mask = detections.tracker_id == track_id
            if np.any(track_mask):
                boxes = detections.xyxy[track_mask]
                confidences = detections.confidence[track_mask]
                classes = detections.class_id[track_mask]
                
                # Focus on rider bounding box specifically
                rider_mask = classes == 0  # Bike_Rider class
                if np.any(rider_mask):
                    # Get rider box(es) for this track
                    rider_boxes = boxes[rider_mask]
                    rider_confidences = confidences[rider_mask]
                    
                    # Use the most confident rider detection or encompass all rider boxes
                    if len(rider_boxes) == 1:
                        min_x, min_y, max_x, max_y = rider_boxes[0]
                    else:
                        # Multiple rider detections - encompass all
                        min_x = np.min(rider_boxes[:, 0])
                        min_y = np.min(rider_boxes[:, 1])
                        max_x = np.max(rider_boxes[:, 2])
                        max_y = np.max(rider_boxes[:, 3])
                else:
                    # Fallback: use all detections if no rider found (shouldn't happen)
                    min_x = np.min(boxes[:, 0])
                    min_y = np.min(boxes[:, 1])
                    max_x = np.max(boxes[:, 2])
                    max_y = np.max(boxes[:, 3])
                
                # Expand rider bounding box slightly for better context
                h, w = frame.shape[:2]
                margin = 30  # Increased margin for better rider context
                min_x = max(0, int(min_x - margin))
                min_y = max(0, int(min_y - margin))
                max_x = min(w, int(max_x + margin))
                max_y = min(h, int(max_y + margin))
                
                # Crop the rider area (violation snapshot)
                violation_crop = frame[min_y:max_y, min_x:max_x]
                
                self.best_violations[track_id] = {
                    'score': violation_score,
                    'frame': violation_crop.copy(),
                    'frame_number': frame_number,
                    'bbox': (min_x, min_y, max_x, max_y),
                    'detections': {
                        'classes': classes.tolist(),
                        'confidences': confidences.tolist(),
                        'boxes': boxes.tolist()
                    }
                }
    
    def save_violation_snapshot(self, track_id, timestamp=None):
        """
        Save the best violation snapshot for a track_id.
        
        Args:
            track_id: Vehicle tracking ID
            timestamp: Optional timestamp for filename
        """
        if track_id not in self.best_violations:
            return
            
        violation_data = self.best_violations[track_id]
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filename = f"violation_track_{track_id}_{timestamp}_score_{violation_data['score']:.2f}.jpg"
        filepath = self.snapshots_dir / filename
        
        cv2.imwrite(str(filepath), violation_data['frame'])
        
        # Save metadata
        metadata = {
            'track_id': int(track_id),
            'violation_score': float(violation_data['score']),
            'frame_number': int(violation_data['frame_number']),
            'bbox': violation_data['bbox'],
            'detections': violation_data['detections'],
            'timestamp': timestamp
        }
        
        metadata_file = self.reports_dir / f"violation_track_{track_id}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved violation snapshot: {filename} (Score: {violation_data['score']:.2f})")
    
    def process_video(self, video_path, output_video_path=None, violation_threshold=0.3):
        """
        Process video for helmet violations.
        
        Args:
            video_path (str): Path to input video
            output_video_path (str): Path to save output video (optional)
            violation_threshold (float): Threshold for flagging violations
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        logger.info(f"Class-specific confidence thresholds: {self.class_confidences}")
        logger.info(f"Violation threshold: {violation_threshold}")
        
        # Setup video writer if output path is provided
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = None
        if output_video_path:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.7)
        
        frame_number = 0
        violations_detected = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_number += 1
                
                # Run inference with very low confidence to catch all potential detections
                # We'll apply class-specific filtering afterward
                results = self.model(frame, conf=0.01)
                
                # Convert to supervision format
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Keep only the 3 classes: Bike_Rider (0), Helmet (1), No_Helmet (2)
                if len(detections) > 0 and detections.class_id is not None:
                    # Keep all 3 classes from the new model
                    class_mask = np.isin(detections.class_id, [0, 1, 2])
                    
                    # Apply the mask to filter detections using supervision's built-in filtering
                    if np.any(class_mask):
                        detections = detections[class_mask]
                        
                        # Apply class-specific confidence thresholds
                        detections = self.apply_class_specific_confidence(detections)
                
                # Update tracker
                detections = self.tracker.update_with_detections(detections)
                
                # Calculate violation scores
                violation_scores = self.calculate_violation_score(detections, detections.xyxy)
                
                # Process each tracked vehicle
                current_violations = []
                for track_id, raw_score in violation_scores.items():
                    # Smooth the violation score
                    smoothed_score = self.smooth_violations(track_id, raw_score)
                    
                    if smoothed_score > violation_threshold:
                        current_violations.append((track_id, smoothed_score))
                        
                        # Update best violation snapshot
                        self.update_best_violation(track_id, smoothed_score, frame, detections, frame_number)
                        
                        # Flag vehicle if not already flagged
                        if track_id not in self.flagged_vehicles:
                            self.flagged_vehicles.add(track_id)
                            violations_detected += 1
                            logger.warning(f"VIOLATION DETECTED - Track ID: {track_id}, Score: {smoothed_score:.2f}, Frame: {frame_number}")
                
                # Create labels for visualization
                labels = []
                for i, (class_id, tracker_id, confidence) in enumerate(zip(detections.class_id, detections.tracker_id, detections.confidence)):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Add violation indicator and priority info
                    violation_indicator = ""
                    priority_marker = ""
                    
                    if class_id == 0:  # Bike_Rider
                        priority_marker = " 🏍️"  # Rider priority marker
                    
                    if tracker_id in violation_scores:
                        smoothed_score = self.smooth_violations(tracker_id, violation_scores[tracker_id])
                        if smoothed_score > violation_threshold:
                            violation_indicator = " ⚠️ VIOLATION"
                    
                    labels.append(f"{class_name} ({confidence:.2f}) ID:{tracker_id}{priority_marker}{violation_indicator}")
                
                # Annotate frame
                annotated_frame = box_annotator.annotate(frame.copy(), detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                
                # Display progress
                if frame_number % 30 == 0:  # Every 30 frames
                    logger.info(f"Processed {frame_number}/{total_frames} frames ({frame_number/total_frames*100:.1f}%)")
                
                # Write frame to output video
                if out:
                    out.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Helmet Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        # Save all violation snapshots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for track_id in self.flagged_vehicles:
            self.save_violation_snapshot(track_id, timestamp)
        
        # Generate summary report
        self.generate_summary_report(video_path, violations_detected, total_frames, timestamp)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total violations detected: {violations_detected}")
        logger.info(f"Output saved to: {self.output_dir}")
        
        return violations_detected
    
    def generate_summary_report(self, video_path, violations_count, total_frames, timestamp):
        """Generate a summary report of the detection results."""
        report = {
            'video_path': str(video_path),
            'timestamp': timestamp,
            'total_frames': int(total_frames),
            'violations_detected': int(violations_count),
            'flagged_vehicles': [int(x) for x in self.flagged_vehicles],
            'violation_details': {}
        }
        
        for track_id in self.flagged_vehicles:
            if track_id in self.best_violations:
                violation_data = self.best_violations[track_id]
                report['violation_details'][int(track_id)] = {
                    'max_violation_score': float(violation_data['score']),
                    'frame_number': int(violation_data['frame_number']),
                    'detection_count': len(violation_data['detections']['classes'])
                }
        
        report_file = self.reports_dir / f"summary_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Helmet Violation Detection System')
    parser.add_argument('--video', type=str, default='aunt.mp4', help='Input video path')
    parser.add_argument('--model', type=str, default='bestt.pt', help='YOLO model path')
    parser.add_argument('--output', type=str, help='Output video path (optional)')
    parser.add_argument('--confidence', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--violation_threshold', type=float, default=0.3, help='Violation threshold')
    parser.add_argument('--tracker', type=str, default='bytetrack', choices=['bytetrack', 'deepsort'], help='Tracker type')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        logger.error(f"Error: Video file '{args.video}' not found!")
        logger.error("Please make sure the video file exists in the current directory.")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Error: Model file '{args.model}' not found!")
        logger.error("Please make sure your trained YOLO model weights are available.")
        return
    
    try:
        # Initialize detector
        detector = HelmetViolationDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            tracker_type=args.tracker
        )
        
        # Process video
        violations = detector.process_video(
            video_path=args.video,
            output_video_path=args.output,
            violation_threshold=args.violation_threshold
        )
        
        if violations > 0:
            logger.warning(f"\n🚨 {violations} helmet violations detected and saved!")
        else:
            logger.info("\n✅ No helmet violations detected.")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()