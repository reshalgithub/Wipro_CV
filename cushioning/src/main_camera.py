import datetime
import math
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import statistics

# -----------------------------
# Configuration
# -----------------------------
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)         # Tracked object
WHITE = (255, 255, 255)     # Text
YELLOW = (0, 255, 255)      # YOLO detections
ROI_COLOR = (255, 0, 0)     # ROI box
DETECTION_INTERVAL = 1

HIGH_SPEED_THRESHOLD = 0.3  # Ratio of max speed
LOW_SPEED_THRESHOLD = 0.001
MIN_CUSHION_FRAMES = 5
SPIKE_IGNORE_RATIO = 0.45    # Ignore speed jumps >45% of neighbors

VIDEO_PATH= r"C:\Users\devteam\Videos\vlc-record-2025-09-03-14h38m06s-vlc-record-2025-09-02-18h19m30s-D04_20250901140722.mp4-.mp4-.mp4"
YOLO_MODEL = r"C:\Users\devteam\Downloads\pumpTopView.pt"
OUTPUT_VIDEO_PATH = "output_piston_tracking.mp4"

# -----------------------------
# Init
# -----------------------------
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=50)

def calculate_speed(prev, current, dt):
    if dt <= 0:
        return 0.0
    dx = current[0] - prev[0]
    dy = current[1] - prev[1]
    return math.sqrt(dx**2 + dy**2) / dt

def smooth_speed(speeds):
    """Remove one-frame spikes caused by camera shake."""
    if len(speeds) < 3:
        return speeds
    smoothed = speeds.copy()
    for i in range(1, len(speeds) - 1):
        prev_s, cur_s, next_s = speeds[i-1], speeds[i], speeds[i+1]
        median_val = statistics.median([prev_s, next_s])
        if cur_s > median_val * (1 + SPIKE_IGNORE_RATIO):
            smoothed[i] = median_val
    return smoothed

def select_roi(video_capture, display_w, display_h):
    ret, first_frame = video_capture.read()
    if not ret:
        raise RuntimeError("Error: Cannot read video file.")
    display_frame = cv2.resize(first_frame, (display_w, display_h))
    roi = cv2.selectROI("Select Object to Track", display_frame, fromCenter=False, showCrosshair=True)
    # Scale ROI back to original frame size
    scale_x = first_frame.shape[1] / display_w
    scale_y = first_frame.shape[0] / display_h
    x_roi = int(roi[0] * scale_x)
    y_roi = int(roi[1] * scale_y)
    w_roi = int(roi[2] * scale_x)
    h_roi = int(roi[3] * scale_y)
    cv2.destroyWindow("Select Object to Track")
    # Reset video to first frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return x_roi, y_roi, w_roi, h_roi

def main():
    video_cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video properties for saving the output
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30  # Default to 30 FPS if not available

    screen_w, screen_h = 1366, 768
    display_scale = min(screen_w / frame_width, screen_h / frame_height)
    display_w, display_h = int(frame_width * display_scale), int(frame_height * display_scale)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    x_roi, y_roi, w_roi, h_roi = select_roi(video_cap, display_w, display_h)

    track_prev = {}
    frame_count = 0
    roi_track_ids = set()  # Track all objects in ROI
    max_observed_speed = {}  # Per track

    cushioning_started = {}  # Per track
    cushion_start_time = {}  # Per track
    cushion_end_time = {}    # Per track
    total_cushion_time = {}  # Per track

    last_yolo_results = []

    while True:
        start_time = datetime.datetime.now()
        frame_count += 1
        ret, frame = video_cap.read()
        if not ret:
            break

        # Draw ROI
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), ROI_COLOR, 2)

        # Define Cushioning Zone (400px from the bottom of the ROI)
        cushion_x1 = x_roi
        cushion_y1 = y_roi + h_roi - 400  # 400px from the bottom
        cushion_x2 = x_roi + w_roi
        cushion_y2 = y_roi + h_roi

        # Draw Cushioning Zone
        cv2.rectangle(frame, (cushion_x1, cushion_y1), (cushion_x2, cushion_y2), (255, 0, 255), 2)  # Magenta color

        run_detection = (frame_count % DETECTION_INTERVAL == 0)
        results = []

        if run_detection:
            detections = model(frame)[0]
            last_yolo_results = []
            for data in detections.boxes.data.tolist():
                confidence = data[4]
                xmin, ymin, xmax, ymax = map(int, data[:4])
                class_id = int(data[5])
                # Show all detections
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), YELLOW, 1)
                cv2.putText(frame, f"{class_id}:{confidence:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

                if confidence >= CONFIDENCE_THRESHOLD:
                    last_yolo_results.append([[xmin, ymin, xmax - xmin, ymax - ymin],
                                    confidence, class_id])
            results = last_yolo_results
        else:
            results = last_yolo_results
            # Optionally, draw cached boxes
            for det in results:
                xmin, ymin, w, h = det[0]
                xmax, ymax = xmin + w, ymin + h
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), YELLOW, 1)
                cv2.putText(frame, f"{det[2]}:{det[1]:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

        tracks = tracker.update_tracks(results, frame=frame)

        # Find all tracks within ROI
        current_roi_tracks = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            xmin, ymin, xmax, ymax = map(int, track.to_ltrb())
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            if (x_roi - 20 < cx < x_roi + w_roi + 20 and
                    y_roi - 20 < cy < y_roi + h_roi + 20):
                current_roi_tracks.add(track.track_id)
                if track.track_id not in roi_track_ids:
                    print(f"New target in ROI: Track ID {track.track_id}")
        
        roi_track_ids = current_roi_tracks

        # Process all tracks in ROI
        for track in tracks:
            if not track.is_confirmed() or track.track_id not in roi_track_ids:
                continue
                
            track_id = track.track_id
            xmin, ymin, xmax, ymax = map(int, track.to_ltrb())
            current_pos = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            current_time = start_time
            speed = 0.0

            if track_id in track_prev:
                prev_time, prev_pos, prev_speeds = track_prev[track_id]
                dt = (current_time - prev_time).total_seconds()
                instant_speed = calculate_speed(prev_pos, current_pos, dt)

                # Add and smooth
                speeds_list = prev_speeds + [instant_speed]
                speeds_list = smooth_speed(speeds_list)[-5:]
                if speeds_list:
                    speed = sum(speeds_list) / len(speeds_list)
            else:
                speeds_list = []

            track_prev[track_id] = (current_time, current_pos, speeds_list)

            # Initialize tracking data for new tracks
            if track_id not in max_observed_speed:
                max_observed_speed[track_id] = 0
                cushioning_started[track_id] = False
                cushion_start_time[track_id] = None
                cushion_end_time[track_id] = None
                total_cushion_time[track_id] = 0.0

            # Max speed tracking per track
            if speed > max_observed_speed[track_id]:
                max_observed_speed[track_id] = speed

            speed_ratio = speed / max_observed_speed[track_id] if max_observed_speed[track_id] > 0 else 0

            # Cushioning detection per track
            # Check if the object is inside the Cushioning Zone
            if cushion_x1 <= cx <= cushion_x2 and cushion_y1 <= cy <= cushion_y2:
                # Cushioning logic applies only within this zone
                if not cushioning_started[track_id] and speed_ratio < HIGH_SPEED_THRESHOLD:
                    if len(speeds_list) >= MIN_CUSHION_FRAMES and all(
                        speeds_list[i] > speeds_list[i+1] for i in range(len(speeds_list)-1)
                    ):
                        cushioning_started[track_id] = True
                        cushion_start_time[track_id] = current_time
                        print(f"Track {track_id}: Cushioning started at {cushion_start_time[track_id]}")

                if cushioning_started[track_id] and cushion_end_time[track_id] is None and speed_ratio < LOW_SPEED_THRESHOLD:
                    cushion_end_time[track_id] = current_time
                    total_cushion_time[track_id] = (cushion_end_time[track_id] - cushion_start_time[track_id]).total_seconds()
                    print(f"Track {track_id}: Cushioning ended | Duration: {total_cushion_time[track_id]:.2f} sec")

            # Draw tracked box with unique colors
            track_id_int = int(track_id) if isinstance(track_id, str) else track_id
            track_color = (0, 255 - (track_id_int * 30) % 256, (track_id_int * 50) % 256)  # Vary colors per track
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), track_color, 2)
            cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 150, ymin), track_color, -1)
            cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (xmin + 5, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            # Show cushioning status per track
            if cushioning_started[track_id]:
                if cushion_end_time[track_id] is None:
                    cv2.putText(frame, f"ID {track_id}: CUSHIONING", (50, 50 + track_id_int * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cushion_text = f"ID {track_id}: Cushioned {total_cushion_time[track_id]:.2f}s"
                    cv2.putText(frame, cushion_text, (50, 50 + track_id_int * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        # FPS
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        fps_text = f"FPS: {1 / time_diff:.2f}" if time_diff > 0 else "FPS: inf"
        cv2.putText(frame, fps_text, (380, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save the frame
        video_writer.write(frame)
        display_frame = cv2.resize(frame, (display_w, display_h))
        cv2.imshow("Piston Tracking", display_frame)
        # Display as fast as possible
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final summary for all tracks
    print("\n=== FINAL SUMMARY ===")
    for track_id in roi_track_ids:
        if track_id in total_cushion_time and total_cushion_time[track_id] > 0:
            print(f"Track {track_id}: Final Cushioning Duration: {total_cushion_time[track_id]:.2f} sec")
        else:
            print(f"Track {track_id}: No cushioning detected")

    video_cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()