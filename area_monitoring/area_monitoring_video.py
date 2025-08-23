import json
import os
import threading
import time
from datetime import datetime
from queue import Empty, Queue

import cv2
import numpy as np
import pygame
from ultralytics import YOLO


class RestrictedAreaMonitor:
    def __init__(
        self,
        model_path="yolov8l.pt",
        camera_id=0,
        save_output=True,
        output_path="output_video.mp4",
    ):
        # Initialize YOLO model
        self.model = YOLO(model_path)

        # Camera setup with optimizations for RTSP
        self.camera_id = camera_id
        self.cap = None
        self.setup_camera()

        # Get video FPS for proper playback speed
        self.video_fps = self.get_video_fps()
        self.frame_time = 1.0 / self.video_fps if self.video_fps > 0 else 1.0 / 30
        print(f"Video FPS: {self.video_fps}, Frame time: {self.frame_time:.4f}s")

        # Video output settings
        self.save_output = save_output
        self.output_path = output_path
        self.video_writer = None
        self.output_initialized = False

        # Threading for frame reading
        self.frame_queue = Queue(maxsize=2)
        self.latest_frame = None
        self.capture_thread = None
        self.running = False

        # Config file for restricted areas
        self.config_file = "restricted_areas.json"

        # Restricted area coordinates (start empty)
        self.restricted_areas = []

        # Track intrusion status for each area
        self.area_intrusion_status = {}  # area_index -> True/False
        self.area_last_intrusion_time = {}  # area_index -> timestamp

        # Load from config if exists
        self.load_areas()

        # Alert settings
        self.alert_active = False
        self.last_alert_time = None
        self.alert_cooldown = 5  # seconds between alerts

        # Initialize pygame for audio alerts
        pygame.mixer.init()

        # Logging setup
        self.log_file = "intrusion_log.json"
        self.alerts_log = []

        # Drawing mode for setting up restricted areas
        self.setup_mode = False
        self.temp_points = []
        self.current_area = []

        # Performance optimization settings
        self.frame_skip = 1  # Reduced from 2 to 1 for smoother playback
        self.frame_count = 0
        self.detection_results = []

        # Timing control
        self.last_frame_time = time.time()

        # Status circle settings
        self.circle_radius = 20
        self.circle_offset_y = -30  # Distance above the area center

    def initialize_video_writer(self, frame_shape):
        """Initialize video writer with frame dimensions"""
        if self.save_output and not self.output_initialized:
            height, width = frame_shape[:2]

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can also use 'XVID'

            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.video_fps, (width, height)
            )

            if self.video_writer.isOpened():
                self.output_initialized = True
                print(
                    f"✅ Video writer initialized: {self.output_path} ({width}x{height} @ {self.video_fps} FPS)"
                )
            else:
                print("❌ Failed to initialize video writer")
                self.save_output = False

    def get_video_fps(self):
        """Get the FPS of the video file"""
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30  # Default to 30 if unable to get FPS
        return 30

    def setup_camera(self):
        """Setup camera with RTSP optimizations"""
        self.cap = cv2.VideoCapture(self.camera_id)

        # RTSP-specific optimizations
        if isinstance(self.camera_id, str) and self.camera_id.startswith("rtsp://"):
            # Use hardware acceleration if available
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4")
            )
            # Buffer settings to reduce lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Network optimizations
            self.cap.set(cv2.CAP_PROP_FPS, 15)

        # For video files, don't override the resolution
        if not (
            isinstance(self.camera_id, str)
            and (
                self.camera_id.startswith("rtsp://")
                or self.camera_id.startswith("http://")
            )
        ):
            # Only set resolution for live cameras, not video files
            if isinstance(self.camera_id, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized with resolution: {actual_width}x{actual_height}")

    def capture_frames(self):
        """Continuously capture frames in separate thread with proper timing"""
        consecutive_failures = 0
        max_failures = 10
        is_video_file = isinstance(
            self.camera_id, str
        ) and self.camera_id.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

        # For video files, we want to respect the original timing
        if is_video_file:
            target_frame_time = self.frame_time
        else:
            target_frame_time = 1.0 / 30  # 30 FPS for live cameras

        last_capture_time = time.time()

        while self.running:
            current_time = time.time()

            # For video files, maintain proper timing
            if is_video_file:
                time_since_last = current_time - last_capture_time
                if time_since_last < target_frame_time:
                    time.sleep(target_frame_time - time_since_last)
                    current_time = time.time()

            ret, frame = self.cap.read()

            if ret:
                consecutive_failures = 0
                last_capture_time = current_time

                # Drop old frames if queue is full
                while self.frame_queue.qsize() >= self.frame_queue.maxsize:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break

                try:
                    self.frame_queue.put_nowait(frame)
                    self.latest_frame = frame
                except Exception as e:
                    print("Queue put error:", e)

            else:
                consecutive_failures += 1

                if is_video_file:
                    print("Reached end of video, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                else:
                    print(
                        f"Failed to read frame (attempt {consecutive_failures}/{max_failures})"
                    )
                    if consecutive_failures >= max_failures:
                        print(
                            "Too many consecutive failures, attempting to reconnect..."
                        )
                        self.reconnect_camera()
                        consecutive_failures = 0
                    time.sleep(0.1)

    def reconnect_camera(self):
        """Reconnect to the camera"""
        if self.cap:
            self.cap.release()

        time.sleep(2)
        self.setup_camera()
        print("Camera reconnection attempted")

    def get_latest_frame(self):
        """Get the most recent frame"""
        frame = None

        # For video files, we want to get frames in order, not skip
        is_video_file = isinstance(
            self.camera_id, str
        ) and self.camera_id.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

        if is_video_file:
            # For video files, get one frame at a time
            try:
                frame = self.frame_queue.get_nowait()
            except Empty:
                frame = self.latest_frame
        else:
            # For live cameras, get the latest frame and discard old ones
            while True:
                try:
                    frame = self.frame_queue.get_nowait()
                except Empty:
                    break

            if frame is None:
                frame = self.latest_frame

        return frame

    def load_areas(self):
        """Load restricted areas from config file if exists"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.restricted_areas = json.load(f)
            print(
                f"Loaded {len(self.restricted_areas)} restricted areas from {self.config_file}"
            )
            # Initialize intrusion status for loaded areas
            for i in range(len(self.restricted_areas)):
                self.area_intrusion_status[i] = False
                self.area_last_intrusion_time[i] = None

    def save_areas(self):
        """Save restricted areas to config file"""
        with open(self.config_file, "w") as f:
            json.dump(self.restricted_areas, f, indent=4)
        print(
            f"Saved {len(self.restricted_areas)} restricted areas to {self.config_file}"
        )

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_area_center(self, area):
        """Calculate the center point of an area"""
        center_x = int(np.mean([p[0] for p in area]))
        center_y = int(np.mean([p[1] for p in area]))
        return (center_x, center_y)

    def get_area_top_point(self, area):
        """Get the topmost point of an area for placing the status circle"""
        min_y = min(p[1] for p in area)
        center_x = int(np.mean([p[0] for p in area]))
        return (center_x, min_y + self.circle_offset_y)

    def draw_status_circles(self, frame):
        """Draw status circles for each restricted area"""
        for i, area in enumerate(self.restricted_areas):
            # Get position for the status circle
            circle_pos = self.get_area_top_point(area)
            
            # Determine circle color based on intrusion status
            is_intruded = self.area_intrusion_status.get(i, False)
            
            if is_intruded:
                # Red circle for intrusion
                circle_color = (0, 0, 255)
                border_color = (0, 0, 200)
                text_color = (255, 255, 255)
                status_text = "ALERT"
            else:
                # Green circle for safe
                circle_color = (0, 255, 0)
                border_color = (0, 200, 0)
                text_color = (0, 0, 0)
                status_text = "SAFE"
            
            # Draw filled circle
            cv2.circle(frame, circle_pos, self.circle_radius, circle_color, -1)
            
            # Draw border
            cv2.circle(frame, circle_pos, self.circle_radius, border_color, 2)
            
            # Add zone number in the circle
            zone_text = str(i + 1)
            text_size = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = circle_pos[0] - text_size[0] // 2
            text_y = circle_pos[1] + text_size[1] // 2
            cv2.putText(
                frame,
                zone_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )
            
            # Add status text below the circle
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            status_x = circle_pos[0] - status_size[0] // 2
            status_y = circle_pos[1] + self.circle_radius + 15
            cv2.putText(
                frame,
                status_text,
                (status_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                border_color,
                1
            )

    def draw_restricted_areas(self, frame):
        """Draw restricted area boundaries on frame"""
        for i, area in enumerate(self.restricted_areas):
            # Draw polygon
            pts = np.array(area, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Determine area color based on intrusion status
            is_intruded = self.area_intrusion_status.get(i, False)
            area_color = (0, 0, 200)

            # Fill area with semi-transparent color
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], area_color)
            alpha = 0.4 if is_intruded else 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw border with different thickness based on status
            border_thickness = 4 if is_intruded else 3
            cv2.polylines(frame, [pts], True, area_color, border_thickness)

            # Add label at the center
            center = self.get_area_center(area)
            label_text = f"RESTRICTED ZONE {i + 1}"
            if is_intruded:
                label_text += " - BREACH!"
                
            cv2.putText(
                frame,
                label_text,
                (center[0] - 80, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def update_area_status(self, intrusions):
        """Update the intrusion status for each area"""
        current_time = datetime.now()
        
        # Reset all areas to safe first
        for i in range(len(self.restricted_areas)):
            self.area_intrusion_status[i] = False
        
        # Mark intruded areas
        for intrusion in intrusions:
            area_idx = intrusion["area_index"]
            self.area_intrusion_status[area_idx] = True
            self.area_last_intrusion_time[area_idx] = current_time

    def check_intrusion(self, detections):
        """Check if any detected person is in restricted area"""
        intrusions = []

        for detection in detections:
            # Get person's bounding box corners
            x1, y1, x2, y2 = detection[:4]
            corners = [
                (int(x1), int(y1)),  # Top-left
                (int(x2), int(y1)),  # Top-right
                (int(x1), int(y2)),  # Bottom-left
                (int(x2), int(y2)),  # Bottom-right
            ]

            # Check against all restricted areas
            for area_idx, area in enumerate(self.restricted_areas):
                # Check if any corner is inside the polygon
                intrusion_detected = False
                for corner in corners:
                    if self.point_in_polygon(corner, area):
                        intrusion_detected = True
                        break

                if intrusion_detected:
                    person_center = (int((x1 + x2) / 2), int(y2))
                    intrusions.append(
                        {
                            "person_bbox": detection,
                            "person_center": person_center,
                            "area_index": area_idx,
                            "timestamp": datetime.now(),
                        }
                    )
                    break

        # Update area status based on current intrusions
        self.update_area_status(intrusions)
        
        return intrusions

    def trigger_alert(self, intrusions):
        """Trigger alert when intrusion is detected"""
        current_time = datetime.now()

        # Check cooldown
        if (
            self.last_alert_time is None
            or (current_time - self.last_alert_time).seconds >= self.alert_cooldown
        ):
            self.alert_active = True
            self.last_alert_time = current_time

            # Log the intrusion
            for intrusion in intrusions:
                log_entry = {
                    "timestamp": intrusion["timestamp"].isoformat(),
                    "area_index": intrusion["area_index"],
                    "person_center": intrusion["person_center"],
                }
                self.alerts_log.append(log_entry)

            # Save to file
            self.save_log()

            # Play alert sound
            try:
                self.play_alert_sound()
            except Exception as e:
                print("Alert sound error:", e)

            # Print alert to console
            print(
                f"🚨 ALERT: {len(intrusions)} person(s) detected in restricted area(s)!"
            )
            for intrusion in intrusions:
                print(
                    f"   - Area {intrusion['area_index'] + 1} at {intrusion['timestamp']}"
                )

    def play_alert_sound(self):
        """Play alert sound"""
        frequency = 1000
        duration = 0.5
        sample_rate = 22050

        frames = int(duration * sample_rate)
        arr = np.zeros(frames)

        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)

        arr = (arr * 32767).astype(np.int16)
        stereo_arr = np.zeros((frames, 2), dtype=np.int16)
        stereo_arr[:, 0] = arr
        stereo_arr[:, 1] = arr

        sound = pygame.sndarray.make_sound(stereo_arr)
        sound.play()

    def save_log(self):
        """Save alerts log to JSON file"""
        with open(self.log_file, "w") as f:
            json.dump(self.alerts_log, f, indent=2)

    def draw_detections(self, frame, detections, intrusions):
        """Draw person detections and highlight intruders"""
        for detection in detections:
            x1, y1, x2, y2, conf = detection[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Check if this person is an intruder
            is_intruder = any(
                np.array_equal(detection[:4], intrusion["person_bbox"][:4])
                for intrusion in intrusions
            )

            # Choose color based on intrusion status
            color = (0, 0, 255) if is_intruder else (0, 255, 0)
            thickness = 3 if is_intruder else -1

            if is_intruder:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw center point
            center = (int((x1 + x2) / 2), int(y2))
            cv2.circle(frame, center, 5, color, -1)

            # Label
            label = f"INTRUDER {conf:.2f}" if is_intruder else ""
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    def setup_restricted_areas(self, event, x, y, flags, param):
        """Mouse callback for setting up restricted areas"""
        if not self.setup_mode:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Point added: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.temp_points) >= 3:
                area_idx = len(self.restricted_areas)
                self.restricted_areas.append(self.temp_points.copy())
                
                # Initialize status for new area
                self.area_intrusion_status[area_idx] = False
                self.area_last_intrusion_time[area_idx] = None
                
                print(
                    f"Restricted area {len(self.restricted_areas)} created with {len(self.temp_points)} points"
                )
                self.save_areas()
                self.temp_points = []
            else:
                print("Need at least 3 points to create a restricted area")

    def run(self):
        """Main monitoring loop with proper frame rate control"""
        print("🎯 Starting Restricted Area Monitoring System")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Setup mode (click to add points, right-click to finish area)")
        print("  'c' - Clear all restricted areas")
        print("  'r' - Reset alert")
        print()

        # Start frame capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        # Set up mouse callback
        cv2.namedWindow("Restricted Area Monitor")
        cv2.setMouseCallback("Restricted Area Monitor", self.setup_restricted_areas)

        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()

        # Frame rate control for display
        is_video_file = isinstance(
            self.camera_id, str
        ) and self.camera_id.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        target_display_time = self.frame_time if is_video_file else 1.0 / 30
        last_display_time = time.time()

        try:
            while True:
                current_time = time.time()

                # Control display frame rate
                time_since_last_display = current_time - last_display_time
                if time_since_last_display < target_display_time:
                    # Wait the remaining time
                    time.sleep(target_display_time - time_since_last_display)
                    current_time = time.time()

                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)  # Small delay if no frame available
                    continue

                last_display_time = current_time

                # Reset alert after some time
                if self.alert_active:
                    if (
                        self.last_alert_time
                        and (datetime.now() - self.last_alert_time).seconds > 3
                    ):
                        self.alert_active = False

                # Process detection every frame for smoother experience
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    # YOLO detection
                    results = self.model(frame, verbose=False)

                    # Filter for person class
                    person_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                if box.cls == 0 and box.conf > 0.5:
                                    person_detections.append(
                                        box.xyxy[0].cpu().numpy().tolist()
                                        + [box.conf.cpu().numpy().item()]
                                    )

                    self.detection_results = person_detections
                else:
                    person_detections = self.detection_results

                # Check for intrusions
                intrusions = self.check_intrusion(person_detections)

                # Trigger alert if intrusions detected
                if intrusions and not self.alert_active:
                    self.trigger_alert(intrusions)

                # Draw everything on frame
                self.draw_restricted_areas(frame)
                self.draw_status_circles(frame)  # Draw status circles
                self.draw_detections(frame, person_detections, intrusions)

                # Draw temporary points in setup mode
                if self.setup_mode:
                    for point in self.temp_points:
                        cv2.circle(frame, point, 5, (255, 255, 0), -1)
                    if len(self.temp_points) > 1:
                        pts = np.array(self.temp_points, np.int32)
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

                    cv2.putText(
                        frame,
                        "SETUP MODE - Click to add points, Right-click to finish area",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # Draw alert status
                if self.alert_active:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "INTRUSION ALERT",
                        (frame.shape[1] // 2 - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3,
                    )

                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    self.current_fps = fps

                # Draw statistics
                stats_y = frame.shape[0] - 80
                cv2.rectangle(frame, (0, stats_y), (450, frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Persons detected: {len(person_detections)}",
                    (10, stats_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Restricted areas: {len(self.restricted_areas)}",
                    (10, stats_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Total alerts: {len(self.alerts_log)}",
                    (10, stats_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if hasattr(self, "current_fps"):
                    cv2.putText(
                        frame,
                        f"FPS: {self.current_fps:.1f}",
                        (250, stats_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow("Restricted Area Monitor", frame)

                # Save frame to output video
                if self.save_output:
                    if not self.output_initialized:
                        self.initialize_video_writer(frame.shape)

                    if self.output_initialized and self.video_writer:
                        self.video_writer.write(frame)

                # Handle keyboard input with minimal delay
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.setup_mode = not self.setup_mode
                    if self.setup_mode:
                        print(
                            "Entered setup mode - click to add points for restricted areas"
                        )
                    else:
                        print("Exited setup mode")
                        if len(self.temp_points) >= 3:
                            area_idx = len(self.restricted_areas)
                            self.restricted_areas.append(self.temp_points.copy())
                            
                            # Initialize status for new area
                            self.area_intrusion_status[area_idx] = False
                            self.area_last_intrusion_time[area_idx] = None
                            
                            print(
                                f"Automatically added last area with {len(self.temp_points)} points"
                            )
                            self.save_areas()
                        self.temp_points = []
                elif key == ord("c"):
                    self.restricted_areas = []
                    self.area_intrusion_status = {}
                    self.area_last_intrusion_time = {}
                    self.save_areas()
                    print("All restricted areas cleared")
                elif key == ord("r"):
                    self.alert_active = False
                    # Reset all area statuses
                    for i in range(len(self.restricted_areas)):
                        self.area_intrusion_status[i] = False
                    print("Alert reset")

        finally:
            # Cleanup
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=2)

            if self.cap and self.cap.isOpened():
                self.cap.release()

            if self.video_writer:
                self.video_writer.release()

            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("✅ Monitoring system stopped.")


if __name__ == "__main__":
    # Initialize and run the monitor
    monitor = RestrictedAreaMonitor(
        model_path="yolo11l.pt",
        camera_id=r"D:\ai_workspace\_Archive\data\HumanDet\VID-20250822-WA0011.mp4",
        save_output=True,  # Enable video saving
        output_path="restricted_area_output.mp4",  # Output video filename
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n⚠️  System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
