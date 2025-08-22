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
DETECTION_INTERVAL = 5

HIGH_SPEED_THRESHOLD = 0.3  # Ratio of max speed
LOW_SPEED_THRESHOLD = 0.001
MIN_CUSHION_FRAMES = 5
SPIKE_IGNORE_RATIO = 0.45    # Ignore speed jumps >30% of neighbors

VIDEO_PATH = r"data\WhatsApp Video 2025-07-31 at 11.53.02.mp4"
YOLO_MODEL = r"E:\AW\Computer Vision\Wipro\model_training\runs01\train\model014\weights\best.pt"
OUTPUT_VIDEO_PATH = "output_piston_tracking.mp4"

# -----------------------------
# Init
# -----------------------------
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=50)

def select_roi(video_capture):
    ret, first_frame = video_capture.read()
    if not ret:
        raise RuntimeError("Error: Cannot read video file.")
    roi = cv2.selectROI("Select Object to Track", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object to Track")
    return roi

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
        if cur_s > median_val * SPIKE_IGNORE_RATIO:
            smoothed[i] = median_val
    return smoothed

def main():
    video_cap = cv2.VideoCapture(VIDEO_PATH)

    # Get video properties for saving the output
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30  # Default to 30 FPS if not available

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    x_roi, y_roi, w_roi, h_roi = select_roi(video_cap)

    track_prev = {}
    frame_count = 0
    selected_track_id = None
    max_observed_speed = 0

    cushioning_started = False
    cushion_start_time = None
    cushion_end_time = None
    total_cushion_time = 0.0

    while True:
        start_time = datetime.datetime.now()
        frame_count += 1
        ret, frame = video_cap.read()
        if not ret:
            break

        # Draw ROI
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), ROI_COLOR, 2)

        run_detection = (selected_track_id is None) or (frame_count % DETECTION_INTERVAL == 0)
        results = []

        if run_detection:
            detections = model(frame)[0]
            for data in detections.boxes.data.tolist():
                confidence = data[4]
                xmin, ymin, xmax, ymax = map(int, data[:4])
                class_id = int(data[5])
                # Show all detections
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), YELLOW, 1)
                cv2.putText(frame, f"{class_id}:{confidence:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

                if confidence >= CONFIDENCE_THRESHOLD:
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin],
                                    confidence, class_id])

        tracks = tracker.update_tracks(results, frame=frame)

        target_track = None
        if selected_track_id is not None:
            for track in tracks:
                if track.track_id == selected_track_id and track.is_confirmed():
                    target_track = track
                    break

        if target_track is None:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                xmin, ymin, xmax, ymax = map(int, track.to_ltrb())
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                if (x_roi - 20 < cx < x_roi + w_roi + 20 and
                        y_roi - 20 < cy < y_roi + h_roi + 20):
                    selected_track_id = track.track_id
                    target_track = track
                    print(f"Acquired new target: Track ID {selected_track_id}")
                    break

        if target_track is not None:
            track_id = target_track.track_id
            xmin, ymin, xmax, ymax = map(int, target_track.to_ltrb())
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

            # Max speed tracking
            if speed > max_observed_speed:
                max_observed_speed = speed

            speed_ratio = speed / max_observed_speed if max_observed_speed > 0 else 0

            # Cushioning detection
            if not cushioning_started and speed_ratio < HIGH_SPEED_THRESHOLD:
                if len(speeds_list) >= MIN_CUSHION_FRAMES and all(
                    speeds_list[i] > speeds_list[i+1] for i in range(len(speeds_list)-1)
                ):
                    cushioning_started = True
                    cushion_start_time = current_time
                    print(f"Cushioning started at {cushion_start_time}")

            if cushioning_started and cushion_end_time is None and speed_ratio < LOW_SPEED_THRESHOLD:
                cushion_end_time = current_time
                total_cushion_time = (cushion_end_time - cushion_start_time).total_seconds()
                print(f"Cushioning ended at {cushion_end_time} | Duration: {total_cushion_time:.2f} sec")

            # Draw tracked box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 150, ymin), GREEN, -1)
            cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (xmin + 5, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            if cushioning_started:
                if cushion_end_time is None:
                    cv2.putText(frame, "CUSHIONING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cushion_text = f"Cushion Time: {total_cushion_time:.2f}s"
                    cv2.putText(frame, cushion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        fps_text = f"FPS: {1 / time_diff:.2f}" if time_diff > 0 else "FPS: inf"
        cv2.putText(frame, fps_text, (380, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save the frame
        video_writer.write(frame)

        cv2.imshow("Piston Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if total_cushion_time > 0:
        print(f"Final Cushioning Duration: {total_cushion_time:.2f} sec")
    else:
        print("Cushioning not detected.")

    video_cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
