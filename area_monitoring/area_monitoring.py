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
    def __init__(self, model_path="yolov8n.pt", camera_id=0):
        # Initialize YOLO model
        self.model = YOLO(model_path)

        # Camera setup with optimizations for RTSP
        self.camera_id = camera_id
        self.cap = None
        self.setup_camera()

        # Threading for frame reading
        self.frame_queue = Queue(maxsize=2)  # Small buffer to reduce lag
        self.latest_frame = None
        self.capture_thread = None
        self.running = False

        # Config file for restricted areas
        self.config_file = "restricted_areas.json"

        # Restricted area coordinates (start empty)
        self.restricted_areas = []

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
        self.frame_skip = 2  # Process every nth frame for detection
        self.frame_count = 0
        self.detection_results = []

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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer

            # Network optimizations
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce bandwidth

        # Resolution settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Additional optimizations
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, 640
        )  # Lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(
            f"Camera initialized with resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )

    def capture_frames(self):
        """Continuously capture frames in separate thread"""
        consecutive_failures = 0
        max_failures = 10

        while self.running:
            ret, frame = self.cap.read()

            if ret:
                consecutive_failures = 0

                # Clear the queue if it's full (drop old frames)
                while self.frame_queue.qsize() >= self.frame_queue.maxsize:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break

                try:
                    self.frame_queue.put_nowait(frame)
                    self.latest_frame = frame
                except Exception as e:
                    print(e)
            else:
                consecutive_failures += 1
                print(
                    f"Failed to read frame (attempt {consecutive_failures}/{max_failures})"
                )

                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures, attempting to reconnect...")
                    self.reconnect_camera()
                    consecutive_failures = 0

                time.sleep(0.1)  # Brief pause before retry

    def reconnect_camera(self):
        """Reconnect to the camera"""
        if self.cap:
            self.cap.release()

        time.sleep(2)  # Wait before reconnecting
        self.setup_camera()
        print("Camera reconnection attempted")

    def get_latest_frame(self):
        """Get the most recent frame, skipping buffered frames"""
        frame = None

        # Get the latest frame from queue, discarding older ones
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except Empty:
                break

        # If no frame from queue, use the latest stored frame
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

    def draw_restricted_areas(self, frame):
        """Draw restricted area boundaries on frame"""
        for i, area in enumerate(self.restricted_areas):
            # Draw polygon
            pts = np.array(area, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Fill area with semi-transparent red
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw border
            cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

            # Add label
            center_x = int(np.mean([p[0] for p in area]))
            center_y = int(np.mean([p[1] for p in area]))
            cv2.putText(
                frame,
                f"RESTRICTED ZONE {i + 1}",
                (center_x - 80, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

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
                (int(x2), int(y2))   # Bottom-right
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
                    person_center = (int((x1 + x2) / 2), int(y2))  # Bottom center for display
                    intrusions.append(
                        {
                            "person_bbox": detection,
                            "person_center": person_center,
                            "area_index": area_idx,
                            "timestamp": datetime.now(),
                        }
                    )
                    break  # No need to check other areas for this detection

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

            # Play alert sound (you can add your own sound file)
            try:
                # Create a simple beep sound
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
        # Generate a simple beep
        frequency = 1000  # Hz
        duration = 0.5  # seconds
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
            color = (
                (0, 0, 255) if is_intruder else (0, 255, 0)
            )  # Red for intruders, green for safe
            thickness = 3 if is_intruder else 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw center point
            center = (int((x1 + x2) / 2), int(y2))
            cv2.circle(frame, center, 5, color, -1)

            # Label
            label = f"INTRUDER {conf:.2f}" if is_intruder else f"Person {conf:.2f}"
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
            if len(self.temp_points) >= 3:  # Need at least 3 points for a polygon
                self.restricted_areas.append(self.temp_points.copy())
                print(
                    f"Restricted area {len(self.restricted_areas)} created with {len(self.temp_points)} points"
                )
                self.save_areas()
                self.temp_points = []
            else:
                print("Need at least 3 points to create a restricted area")

    def run(self):
        """Main monitoring loop"""
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

        try:
            while True:
                frame = self.get_latest_frame()
                if frame is None:
                    continue

                # Reset alert after some time
                if self.alert_active:
                    current_time = datetime.now()
                    if (
                        self.last_alert_time
                        and (current_time - self.last_alert_time).seconds > 3
                    ):
                        self.alert_active = False

                # Skip frames for detection to improve performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    # YOLO detection (only on every nth frame)
                    results = self.model(frame, verbose=False)

                    # Filter for person class (class 0 in COCO dataset)
                    person_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                if (
                                    box.cls == 0 and box.conf > 0.5
                                ):  # Person class with confidence > 0.5
                                    person_detections.append(
                                        box.xyxy[0].cpu().numpy().tolist()
                                        + [box.conf.cpu().numpy().item()]
                                    )

                    self.detection_results = person_detections
                else:
                    # Use previous detection results for frames in between
                    person_detections = self.detection_results

                # Check for intrusions
                intrusions = self.check_intrusion(person_detections)

                # Trigger alert if intrusions detected
                if intrusions and not self.alert_active:
                    self.trigger_alert(intrusions)

                # Draw everything on frame
                self.draw_restricted_areas(frame)
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
                if fps_counter % 30 == 0:  # Calculate FPS every 30 frames
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

                # Handle keyboard input
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
                            self.restricted_areas.append(self.temp_points.copy())
                            print(
                                f"Automatically added last area with {len(self.temp_points)} points for monitoring"
                            )
                            self.save_areas()
                        self.temp_points = []
                elif key == ord("c"):
                    self.restricted_areas = []
                    self.save_areas()
                    print("All restricted areas cleared")
                elif key == ord("r"):
                    self.alert_active = False
                    print("Alert reset")

        finally:
            # Cleanup
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=2)

            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("✅ System shutdown complete")


if __name__ == "__main__":
    # Initialize and run the monitor
    monitor = RestrictedAreaMonitor(
        model_path="yolov8n.pt",  # Download automatically if not present
        camera_id="rtsp://devteam:pctadmin@172.36.0.131:554/stream1",  # Your RTSP stream
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n⚠️  System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
