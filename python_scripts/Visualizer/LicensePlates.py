#!/usr/bin/env python3
"""
License Plate Detection and Tracking Module
Integrates with existing video analysis system to detect, track, and store license plate information.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import defaultdict
from pathlib import Path
import easyocr
import re
from datetime import datetime

class LicensePlateDetector:
    def __init__(self, device="auto", confidence_threshold=0.3):
        """
        Initialize license plate detector with YOLO and OCR
        
        Args:
            device: 'auto', 'cpu', or 'cuda'
            confidence_threshold: Minimum confidence for detection
        """
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[INFO] License Plate Detector using device: {self.device}")
        
        # Load YOLO model (will use general model and filter for cars/vehicles)
        self.model = YOLO("yolo11x.pt").to(self.device)
        
        # Initialize EasyOCR for license plate text recognition
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
        
        # Setup tracker for license plates
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=60  # Higher threshold for license plates
        )
        
        # Tracking history
        self.plate_history = defaultdict(list)
        self.confidence_threshold = confidence_threshold
        
        # Vehicle class IDs that might have license plates
        self.vehicle_classes = {
            'car': 2,
            'motorcycle': 3, 
            'bus': 5,
            'truck': 7
        }

    def is_likely_license_plate_region(self, bbox, frame_shape):
        """
        Heuristic to determine if a region might contain a license plate
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            bool: True if region might contain a license plate
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # License plates are typically wider than they are tall
        aspect_ratio = width / height if height > 0 else 0
        
        # Typical license plate aspect ratios: 2:1 to 5:1
        if not (1.5 <= aspect_ratio <= 6.0):
            return False
            
        # Size constraints - not too small or too large relative to frame
        frame_area = frame_shape[0] * frame_shape[1]
        region_area = width * height
        area_ratio = region_area / frame_area
        
        # License plates should be between 0.01% and 5% of frame area
        if not (0.0001 <= area_ratio <= 0.05):
            return False
            
        return True

    def extract_license_plate_regions(self, frame, vehicle_detections):
        """
        Extract potential license plate regions from detected vehicles
        
        Args:
            frame: Video frame
            vehicle_detections: List of vehicle bounding boxes
            
        Returns:
            List of potential license plate regions with coordinates
        """
        plate_regions = []
        
        for vehicle_bbox in vehicle_detections:
            x1, y1, x2, y2 = vehicle_bbox
            
            # Focus on front and rear areas of vehicles for license plates
            vehicle_width = x2 - x1
            vehicle_height = y2 - y1
            
            # Front region (bottom portion of vehicle)
            front_region = [
                x1 + int(vehicle_width * 0.1),  # Leave some margin
                y2 - int(vehicle_height * 0.3),  # Bottom 30% of vehicle
                x2 - int(vehicle_width * 0.1),
                y2
            ]
            
            # Rear region (if visible, similar logic)
            rear_region = [
                x1 + int(vehicle_width * 0.1),
                y1,
                x2 - int(vehicle_width * 0.1),
                y1 + int(vehicle_height * 0.3)  # Top 30% of vehicle
            ]
            
            # Add regions that pass our heuristics
            for region in [front_region, rear_region]:
                if self.is_likely_license_plate_region(region, frame.shape):
                    plate_regions.append(region)
                    
        return plate_regions

    def perform_ocr_on_region(self, frame, region):
        """
        Perform OCR on a specific region to extract license plate text
        
        Args:
            frame: Video frame
            region: Bounding box [x1, y1, x2, y2]
            
        Returns:
            tuple: (license_plate_text, confidence)
        """
        x1, y1, x2, y2 = region
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
            
        # Extract region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0.0
            
        # Preprocess for better OCR
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        enhanced_roi = cv2.convertScaleAbs(gray_roi, alpha=1.5, beta=30)
        
        # Apply slight blur to reduce noise
        blurred_roi = cv2.GaussianBlur(enhanced_roi, (3, 3), 0)
        
        try:
            # Perform OCR
            results = self.ocr_reader.readtext(blurred_roi)
            
            if not results:
                return None, 0.0
                
            # Find the most confident result that looks like a license plate
            best_text = None
            best_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                # Clean up text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # License plate heuristics
                if (3 <= len(cleaned_text) <= 10 and  # Reasonable length
                    confidence > 0.4 and  # Minimum confidence
                    any(c.isalnum() for c in cleaned_text)):  # Contains alphanumeric
                    
                    if confidence > best_confidence:
                        best_text = cleaned_text
                        best_confidence = confidence
                        
            return best_text, best_confidence
            
        except Exception as e:
            print(f"[WARN] OCR failed on region: {e}")
            return None, 0.0

    def detect_license_plates_in_frame(self, frame, frame_idx, second, gps_data):
        """
        Detect license plates in a single frame
        
        Args:
            frame: Video frame
            frame_idx: Frame index
            second: Time in seconds
            gps_data: GPS data dictionary
            
        Returns:
            List of license plate detections
        """
        # Get GPS data for this second
        cam_lat, cam_lon = gps_data.get(second, (None, None))
        
        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]
        
        # Extract vehicle detections
        vehicle_detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            class_id = int(cls)
            if class_id in self.vehicle_classes.values():
                vehicle_detections.append(box.cpu().numpy())
        
        # Get potential license plate regions
        plate_regions = self.extract_license_plate_regions(frame, vehicle_detections)
        
        # Perform OCR on each region
        detections = []
        for region in plate_regions:
            license_text, ocr_confidence = self.perform_ocr_on_region(frame, region)
            
            if license_text and ocr_confidence > 0.5:  # Only high-confidence plates
                x1, y1, x2, y2 = region
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detection = Detection(
                    points=np.array([[center_x, center_y]]),
                    scores=np.array([ocr_confidence])
                )
                
                # Add metadata
                detection.license_text = license_text
                detection.bbox = region
                detection.frame_idx = frame_idx
                detection.second = second
                detection.cam_lat = cam_lat
                detection.cam_lon = cam_lon
                detection.ocr_confidence = ocr_confidence
                
                detections.append(detection)
        
        return detections

    def compute_direction_of_travel(self, track_history):
        """
        Compute direction of travel from tracking history
        
        Args:
            track_history: List of (x, y) positions over time
            
        Returns:
            float: Direction in degrees (0-360)
        """
        if len(track_history) < 2:
            return None
            
        # Use first and last points for overall direction
        start_point = np.array(track_history[0])
        end_point = np.array(track_history[-1])
        
        delta = end_point - start_point
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = (np.degrees(angle_rad) + 360) % 360
        
        return angle_deg

    def process_video_for_license_plates(self, video_path, gps_data, output_dir):
        """
        Process entire video for license plate detection and tracking
        
        Args:
            video_path: Path to video file
            gps_data: GPS data dictionary (second -> (lat, lon))
            output_dir: Output directory path
            
        Returns:
            str: Path to output CSV file
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Create license plates folder
        license_plates_dir = output_dir / "license_plates"
        license_plates_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Processing {video_path.name} for license plates...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        license_plate_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            second = int(frame_idx / fps)
            
            # Detect license plates in frame
            detections = self.detect_license_plates_in_frame(frame, frame_idx, second, gps_data)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Process tracked license plates
            for tracked_obj in tracked_objects:
                if hasattr(tracked_obj, 'last_detection') and hasattr(tracked_obj.last_detection, 'license_text'):
                    detection = tracked_obj.last_detection
                    track_id = tracked_obj.id
                    
                    # Store position in history
                    center_point = detection.points[0]
                    self.plate_history[track_id].append(center_point)
                    
                    # Compute direction of travel
                    direction = self.compute_direction_of_travel(self.plate_history[track_id])
                    
                    # Create record
                    record = {
                        'license_plate': detection.license_text,
                        'coordinates_x': float(center_point[0]),
                        'coordinates_y': float(center_point[1]),
                        'direction_of_travel': direction,
                        'frame': frame_idx,
                        'second': second,
                        'confidence': detection.ocr_confidence,
                        'cam_lat': detection.cam_lat,
                        'cam_lon': detection.cam_lon,
                        'track_id': track_id
                    }
                    
                    license_plate_data.append(record)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[INFO] Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        # Save results to CSV
        output_csv = license_plates_dir / f"{video_path.stem}_licenseplates.csv"
        
        if license_plate_data:
            df = pd.DataFrame(license_plate_data)
            
            # Group by license plate and keep best detections
            df_grouped = df.groupby('license_plate').agg({
                'coordinates_x': 'mean',
                'coordinates_y': 'mean', 
                'direction_of_travel': 'first',
                'frame': 'first',
                'second': 'first',
                'confidence': 'max',
                'cam_lat': 'first',
                'cam_lon': 'first',
                'track_id': 'first'
            }).reset_index()
            
            # Final output columns as requested
            final_df = df_grouped[['license_plate', 'coordinates_x', 'coordinates_y', 'direction_of_travel']].copy()
            final_df.columns = ['license_plate', 'coordinates_seen_at_x', 'coordinates_seen_at_y', 'direction_of_travel']
            
            final_df.to_csv(output_csv, index=False)
            print(f"[INFO] Saved license plate data: {output_csv}")
            print(f"[INFO] Found {len(final_df)} unique license plates")
            
        else:
            # Create empty CSV with correct columns
            empty_df = pd.DataFrame(columns=['license_plate', 'coordinates_seen_at_x', 'coordinates_seen_at_y', 'direction_of_travel'])
            empty_df.to_csv(output_csv, index=False)
            print(f"[INFO] No license plates detected in {video_path.name}")
        
        return str(output_csv)


def load_gps_data(csv_path):
    """Load GPS data from merged_output.csv"""
    try:
        df = pd.read_csv(csv_path)
        return {int(row['second']): (row['lat'], row['long']) for _, row in df.iterrows()}
    except Exception as e:
        print(f"[ERROR] Failed to load GPS data: {e}")
        return {}


def process_video_directory(video_dir):
    """
    Process all videos in a directory for license plate detection
    
    Args:
        video_dir: Directory containing videos and merged_output.csv
    """
    video_dir = Path(video_dir)
    
    # Find video files
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))
    
    if not video_files:
        print(f"[ERROR] No video files found in {video_dir}")
        return
    
    # Load GPS data
    gps_csv = video_dir / "merged_output.csv"
    if not gps_csv.exists():
        print(f"[ERROR] merged_output.csv not found in {video_dir}")
        return
    
    gps_data = load_gps_data(gps_csv)
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Process each video
    all_csv_files = []
    for video_file in video_files:
        print(f"\n[INFO] Processing video: {video_file.name}")
        
        csv_file = detector.process_video_for_license_plates(
            video_file, gps_data, video_dir
        )
        
        if csv_file:
            all_csv_files.append(csv_file)
    
    # Combine all CSVs into a master file
    if all_csv_files:
        combine_all_license_plate_csvs(all_csv_files, video_dir / "license_plates")
    
    print(f"\n[INFO] License plate processing complete!")


def combine_all_license_plate_csvs(csv_files, license_plates_dir):
    """
    Combine all individual license plate CSVs into a master CSV
    
    Args:
        csv_files: List of CSV file paths
        license_plates_dir: Directory containing license plate CSVs
    """
    combined_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Add source video column
                video_name = Path(csv_file).stem.replace('_licenseplates', '')
                df['source_video'] = video_name
                combined_data.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_file}: {e}")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        output_path = license_plates_dir / "all_license_plates_combined.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"[INFO] Combined license plate data saved: {output_path}")
        print(f"[INFO] Total unique license plates across all videos: {len(combined_df)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="License Plate Detection and Tracking")
    parser.add_argument("-d", "--directory", required=True, 
                        help="Directory containing video files and merged_output.csv")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    # Process the directory
    process_video_directory(args.directory)