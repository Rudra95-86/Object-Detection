import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, detection_model='yolo11x.pt', target_class=0, use_gpu=True, fullscreen=True):
        """
        Initialize the Object Tracker with YOLOv11.
        """
        # Try to find Droidcam camera 
        cam_index = -1
        for i in range(5):  # check first 5 indexes
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                ret, frame = cap_test.read()
                if ret:
                    print(f"Camera index {i} available")
                    if frame.shape[1] >= 1280:  
                        cam_index = i
                        print(f"Using camera index {i} (likely Droidcam)")
                cap_test.release()

        if cam_index == -1:
            cam_index = 0  
            print("Camo Studio not detected, falling back to default webcam")

        # Initialize video capture with the detected index
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
                    
        # Set camera properties for higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # height
        self.cap.set(cv2.CAP_PROP_FPS, 90)

        # Fullscreen mode
        self.fullscreen = fullscreen
        
        # Initialize YOLOv11 model
        self.detection_model = YOLO(detection_model)
        self.target_class = target_class
        
        # Initialize OpenCV trackers - check which tracker functions are available
        self.opencv_trackers = {}
        
        # Try to initialize available trackers
        try:
            tracker = cv2.TrackerKCF_create()
            self.opencv_trackers['KCF'] = cv2.TrackerKCF_create
            print("KCF tracker available")
        except:
            print("KCF tracker not available")
            
        try:
            tracker = cv2.TrackerCSRT_create()
            self.opencv_trackers['CSRT'] = cv2.TrackerCSRT_create
            print("CSRT tracker available")
        except:
            print("CSRT tracker not available")
            
        try:
            tracker = cv2.TrackerMIL_create()
            self.opencv_trackers['MIL'] = cv2.TrackerMIL_create
            print("MIL tracker available")
        except:
            print("MIL tracker not available")
       
        if not self.opencv_trackers:
            raise ValueError("No OpenCV trackers available!")
            
        self.current_tracker_type = list(self.opencv_trackers.keys())[0]
        self.trackers = []  # List of (tracker_id, tracker, label, color)
        self.next_tracker_id = 1
        
        # Tracking mode: 'classical' or 'deep_learning'
        self.mode = 'deep_learning' 
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Colors for different tracks - more colors for more objects
        self.colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
        
        # Mouse event handling for ROI selection
        self.drawing_box = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        
        # For deep learning tracking
        self.track_history = defaultdict(lambda: [])
        
        print(f"ObjectTracker initialized with {list(self.opencv_trackers.keys())} trackers.")
        print("Press 's' to select ROI, 'd' for YOLO mode, 'c' to clear, 'q' to quit.")
        print("Press 'f' to toggle fullscreen mode")
    
    def select_roi(self, event, x, y, flags, param):
        """
        Mouse callback function for selecting Region of Interest (ROI).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_box = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_box = False
            self.end_point = (x, y)
            
            # Create a tracker for the selected ROI
            if self.start_point != self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                bbox = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                
                # Initialize tracker
                try:
                    tracker = self.opencv_trackers[self.current_tracker_type]()
                    tracker.init(self.current_frame, bbox)
                    
                    # Add to trackers list
                    tracker_id = self.next_tracker_id
                    self.next_tracker_id += 1
                    color = tuple(map(int, self.colors[tracker_id % len(self.colors)]))
                    self.trackers.append((tracker_id, tracker, "Object", color))
                    
                    print(f"Added tracker {tracker_id} with {self.current_tracker_type}")
                except Exception as e:
                    print(f"Error initializing tracker: {e}")
    
    def run_classical_tracking(self, frame):
        """
        Update all classical trackers and draw results.
        """
        updated_trackers = []
        
        for tracker_id, tracker, label, color in self.trackers:
            try:
                success, bbox = tracker.update(frame)
                
                if success:
                    # Draw bounding box
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label_text = f"{label} {tracker_id}"
                    cv2.putText(frame, label_text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    updated_trackers.append((tracker_id, tracker, label, color))
                else:
                    print(f"Tracker {tracker_id} lost target")
            except Exception as e:
                print(f"Error updating tracker {tracker_id}: {e}")
        
        self.trackers = updated_trackers
        return frame
    
    def run_deep_learning_tracking(self, frame):
        """
        Perform YOLOv11 detection and tracking for multiple objects.
        """
        try:
            # Run YOLOv11 inference with tracking
            results = self.detection_model.track(
                frame, 
                persist=True, 
                verbose=False,
                conf=0.5,  # Lower confidence threshold to detect more objects
                imgsz=640  # Larger image size for better detection
            )
            
            # Check if results exist and have boxes
            if results and results[0].boxes is not None:
                # Get boxes, track IDs, and class IDs
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Get track IDs if available
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    # If no track IDs, create them based on detection index
                    track_ids = np.arange(len(boxes))
                
                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(class_ids[i])
                    confidence = confidences[i]
                    track_id = int(track_ids[i])
                    
                    # Only draw if confidence is high enough
                    if confidence > 0.3:  # Lower threshold to track more objects
                        # Get color for this track ID
                        color = tuple(map(int, self.colors[track_id % len(self.colors)]))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with background for better visibility
                        label = self.detection_model.names[class_id]
                        label_text = f"{label} {track_id} {confidence:.2f}"
                        
                        # Draw background for text
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(frame, label_text, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error in deep learning tracking: {e}")
        
        return frame
    
    def calculate_fps(self):
        """Calculate and display FPS."""
        self.frame_count += 1
        if self.frame_count >= 30:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
            self.frame_count = 0
            
        return self.fps
    
    def run(self):
        """Main loop for object tracking."""
        window_name = 'Object Tracking - Press F for fullscreen, Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size to fullscreen or large size
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720) 
        
        cv2.setMouseCallback(window_name, self.select_roi)
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            self.current_frame = frame.copy()
            
            # Apply tracking based on mode
            if self.mode == 'classical':
                frame = self.run_classical_tracking(frame)
                mode_text = f"Mode: Classical ({self.current_tracker_type})"
            else:
                frame = self.run_deep_learning_tracking(frame)
                mode_text = "Mode: Deep Learning (YOLOv11)"
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display current mode
            cv2.putText(frame, mode_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display object count
            if self.mode == 'deep_learning' and hasattr(self, 'current_frame'):
                try:
                    results = self.detection_model.track(self.current_frame, persist=True, verbose=False)
                    if results and results[0].boxes is not None:
                        count = len(results[0].boxes)
                        cv2.putText(frame, f"Objects: {count}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except:
                    pass
            
            # Display available trackers
            tracker_text = f"Trackers: {', '.join(self.opencv_trackers.keys())}"
            cv2.putText(frame, tracker_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display instructions
            instructions = [
                "Press 's': Select ROI",
                "Press 'd': Switch mode",
                "Press 'c': Clear tracks",
                "Press 'f': Toggle fullscreen",
                "Press 'q': Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 30 - i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw ROI selection box if in progress
            if self.drawing_box and self.start_point != (-1, -1) and self.end_point != (-1, -1):
                cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.mode = 'classical'
                print("Switched to classical mode. Select ROI with mouse.")
            elif key == ord('d'):
                self.mode = 'deep_learning'
                print("Switched to deep learning mode.")
            elif key == ord('c'):
                self.trackers = []
                print("All trackers cleared.")
            elif key == ord('k'):
                # Switch between available trackers
                tracker_types = list(self.opencv_trackers.keys())
                current_index = tracker_types.index(self.current_tracker_type)
                next_index = (current_index + 1) % len(tracker_types)
                self.current_tracker_type = tracker_types[next_index]
                print(f"Switched to {self.current_tracker_type} tracker")
            elif key == ord('f'):
                # Toggle fullscreen
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Switched to fullscreen mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)
                    print("Switched to windowed mode")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Available YOLOv11 models: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    tracker = ObjectTracker(detection_model='yolo11x.pt', fullscreen=True)
    tracker.run()