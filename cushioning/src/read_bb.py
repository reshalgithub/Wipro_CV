import datetime
import math
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# -----------------------------
# Configuration
# -----------------------------
CONFIDENCE_THRESHOLD = 0.8   # YOLO confidence threshold
GREEN = (0, 255, 0)          # Bounding box color
WHITE = (255, 255, 255)      # Text color
YELLOW = (0, 255, 255)      # YOLO detections
ROI_COLOR = (255, 0, 0)     # ROI box
DETECTION_INTERVAL = 5      # After acquisition, run YOLO every N frames

VIDEO_PATH = r"data\WhatsApp Video 2025-07-31 at 11.53.02.mp4"
# VIDEO_PATH = r"data\WhatsApp Video 2025-07-31 at 11.52.54.mp4"
YOLO_MODEL = r"E:\AW\Computer Vision\Wipro\model_training\runs01\train\model014\weights\best.pt"
# YOLO_MODEL = "yolov8n.pt"

# -----------------------------
# Initialize YOLO + Tracker
# -----------------------------
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=50)


def select_roi(video_capture):
    ret, first_frame = video_capture.read()
    if not ret:
        raise RuntimeError("Error: Cannot read video file.")
    roi = cv2.selectROI("Select Object to Track", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object to Track")
    return roi  # (x, y, w, h)

def calculate_speed(prev, current, dt):
    # speed in pixels/second
    if dt <= 0:
        return 0.0
    dx = current[0] - prev[0]
    dy = current[1] - prev[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance / dt

def main():
    video_cap = cv2.VideoCapture(VIDEO_PATH)
    x_roi, y_roi, w_roi, h_roi = select_roi(video_cap)

    track_prev = {}
    frame_count = 0
    selected_track_id = None

    while True:
        start_time = datetime.datetime.now()
        frame_count += 1
        ret, frame = video_cap.read()
        if not ret:
            break

        # Draw ROI rectangle
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), ROI_COLOR, 2)

        # Run YOLO more frequently before acquisition
        run_detection = (selected_track_id is None) or (frame_count % DETECTION_INTERVAL == 0)
        results = []

        if run_detection:
            detections = model(frame)[0]
            for data in detections.boxes.data.tolist():
                confidence = data[4]
                xmin, ymin, xmax, ymax = map(int, data[:4])
                class_id = int(data[5])
                # Draw ALL detections
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
            # Try to find a new target in ROI
            reason = "No tracks in ROI"
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
                    reason = f"Acquired new target ID {selected_track_id}"
                    break
            print(reason)

        # Draw tracked target + calculate speed
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
                new_speeds = (prev_speeds + [instant_speed])[-5:]
                if new_speeds:
                    speed = sum(new_speeds) / len(new_speeds)
            else:
                new_speeds = []

            track_prev[track_id] = (current_time, current_pos, new_speeds)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 150, ymin), GREEN, -1)
            cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (xmin + 5, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # FPS display
        end_time = datetime.datetime.now()
        time_diff = (end_time - start_time).total_seconds()
        fps_text = f"FPS: {1 / time_diff:.2f}" if time_diff > 0 else "FPS: inf"
        cv2.putText(frame, fps_text, (380, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Tracking Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()