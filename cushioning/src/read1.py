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
DETECTION_INTERVAL = 5       # Run detector every N frames to improve performance

VIDEO_PATH = r"data\WhatsApp Video 2025-07-31 at 11.53.02.mp4"
# VIDEO_PATH = r"data\WhatsApp Video 2025-07-31 at 11.52.54.mp4"
# YOLO_MODEL = "yolov8n.pt"
YOLO_MODEL = r"E:\AW\Computer Vision\Wipro\model_training\runs01\train\model014\weights\best.pt"

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

    # Step 1: User selects object in first frame
    x_roi, y_roi, w_roi, h_roi = select_roi(video_cap)

    track_prev = {} # For storing previous positions
    selected_track_id = None

    while True:
        start_time = datetime.datetime.now()
        ret, frame = video_cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        results = []
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = map(int, data[:4])
            class_id = int(data[5])

            if selected_track_id is None:
                if (xmin > x_roi and ymin > y_roi and
                        xmax < x_roi + w_roi and ymax < y_roi + h_roi):
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin],
                                    confidence, class_id])
            else:
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin],
                                confidence, class_id])

        # Update DeepSort tracker
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            xmin, ymin, xmax, ymax = map(int, track.to_ltrb())

            # Assign the selected object?s track_id
            if selected_track_id is None:
                selected_track_id = track_id

            if track_id != selected_track_id:
                continue

            # Calculate speed
            current_pos = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            current_time = start_time

            if track_id in track_prev:
                prev_time, prev_pos = track_prev[track_id]
                dt = (current_time - prev_time).total_seconds()
                speed = calculate_speed(prev_pos, current_pos, dt)
            else:
                speed = 0.0

            track_prev[track_id] = (current_time, current_pos)
            # Draw bounding box and info
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 150, ymin), GREEN, -1)
            cv2.putText(frame, f"ID: {track_id}", (xmin + 5, ymin - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (xmin + 5, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        # FPS display
        end_time = datetime.datetime.now()
        fps_text = f"FPS: {1 / (end_time - start_time).total_seconds():.2f}"
        cv2.putText(frame, fps_text, (380, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()