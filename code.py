import math
import cv2
from ultralytics import YOLO

# 1. Load the YOLO model and video file
# (Replace "yolov8n.pt" with your custom model path if you have one)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("vechile_video.mp4")

# Get the video's FPS (Frames Per Second)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or math.isnan(fps):
    fps = 30  # Default fallback FPS

# 2. Calibration setting (Adjust this value based on your camera view)
# Assume 10 pixels on the road equals 1 real-world meter
PIXELS_PER_METER = 10.0

# COCO class IDs for vehicles (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Dictionaries to store each vehicle's previous position and smoothed speed
prev_positions = {}
vehicle_speeds = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLOv8 object tracking (persist=True keeps IDs consistent across frames)
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls_id]

            # Calculate the center point of the vehicle
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 4. Calculate vehicle speed
            if track_id in prev_positions:
                prev_cx, prev_cy = prev_positions[track_id]

                # Calculate pixel distance moved between frames
                pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)

                # Convert pixel distance to meters
                distance_meters = pixel_distance / PIXELS_PER_METER

                # Speed (m/s) = distance / time (where time = 1 / fps)
                speed_mps = distance_meters * fps

                # Convert m/s to km/h (1 m/s = 3.6 km/h)
                speed_kmh = speed_mps * 3.6

                # Smooth the speed reading using an exponential moving average
                if track_id in vehicle_speeds:
                    speed_kmh = 0.7 * vehicle_speeds[track_id] + 0.3 * speed_kmh

                vehicle_speeds[track_id] = speed_kmh
            else:
                vehicle_speeds[track_id] = 0.0

            # Save current position for the next frame
            prev_positions[track_id] = (cx, cy)

            # 5. Draw bounding box, center dot, ID, and speed on the frame
            speed_text = f"ID:{track_id} {label} | {int(vehicle_speeds[track_id])} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                speed_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Vehicle Speed Detection", frame)

    # Press 'Esc' (27) or 'q' to exit the video window
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()