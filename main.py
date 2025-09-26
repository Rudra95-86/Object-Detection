import argparse
from tracker_comparison import ObjectTracker

def main():
    parser = argparse.ArgumentParser(description='Real-Time Object Tracking in Video Streams')
    parser.add_argument('--detection_model', type=str, default='yolo11x.pt',
                        help='Path to YOLOv8 model weights (default: yolo11x.pt)')
    parser.add_argument('--target_class', type=int, default=0,
                        help='COCO class ID to track (0 for person, default: 0)')
    
    args = parser.parse_args()
    
    # Create and run the tracker
    try:
        tracker = ObjectTracker(
            detection_model=args.detection_model,
            target_class=args.target_class
        )
        tracker.run()
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Make sure you have a webcam connected and OpenCV is properly installed.")
    finally:
        print("Exiting...")
        

if __name__ == "__main__":
    main()