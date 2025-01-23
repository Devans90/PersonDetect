import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pydantic import BaseModel
from ultralytics import YOLO

# ---------------------------
# 1. Define data structures
# ---------------------------

class ROI(BaseModel):
    name: str
    # We'll define ROIs in normalized coordinates: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    # Each coordinate is in [0, 1] range, so that the ROI adjusts with the frame size.
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    
    # We'll keep both an instantaneous detection count (for a single frame)
    # and a cumulative detection count (for the entire video).
    frame_attention_count: int = 0
    total_attention_count: int = 0
    
    # Define a unique BGR color for each stand overlay
    color_bgr: Tuple[int, int, int] = (255, 200, 100)  # default color (light blue)

# Define your ROIs in normalized coordinates
rois = [
    ROI(name="Stand 1",
        top_left=(0.1, 0.05),
        bottom_right=(0.30, 0.60),
        color_bgr=(255, 200, 100)),  # Light blue
    ROI(name="Stand 2",
        top_left=(0.4, 0.05),
        bottom_right=(0.80, 0.40),
        color_bgr=(100, 200, 255)),  # Light orange-ish
    # Add more stands as needed, with unique colors
]

# Overlap threshold (i.e., require at least 30% overlap between
# the person's bounding box and the ROI to consider them "in" the stand)
OVERLAP_THRESHOLD = 0.3

# ---------------------------
# 2. Initialize Model
# ---------------------------

model = YOLO('yolo11n.pt')  # or your chosen YOLO version
# You can play around with model parameters, e.g.:
# results = model(frame, conf=0.25, iou=0.45)

# ---------------------------
# 3. Define helper functions
# ---------------------------

def compute_intersection_area(boxA, boxB):
    """
    Compute the intersection area between two bounding boxes.
    boxA and boxB are each in (x1, y1, x2, y2) format (pixel coords).
    """
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    # Calculate overlap area
    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0  # No overlap

    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

def compute_area(box):
    """Compute area of a bounding box in (x1, y1, x2, y2) format."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def smooth_data(data_list, window_size=5):
    """
    Simple moving average smoothing for 1D data.
    data_list: list of values
    window_size: number of frames to average
    Returns a new list of the same length, smoothed by the window.
    """
    smoothed = []
    for i in range(len(data_list)):
        start = max(0, i - window_size + 1)
        window = data_list[start:i+1]
        smoothed.append(sum(window) / len(window))
    return smoothed

# ---------------------------
# 4. Video Capture
# ---------------------------

cap = cv2.VideoCapture('cctv_footage.mp4')

# We store historical data for time-series:
# Each entry will track "frame", each stand's count, and "Outside".
timeline_data = []

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    height, width = frame.shape[:2]

    # Reset each ROI's frame-based count
    for roi in rois:
        roi.frame_attention_count = 0

    # We'll track how many people were not in any stand for this frame
    outside_count = 0

    # 1) Predict with YOLO (person detection)
    results = model(frame)

    # 2) Prepare an overlay copy for semi-transparent ROI rectangles
    overlay = frame.copy()
    for roi in rois:
        roi_x1 = int(roi.top_left[0] * width)
        roi_y1 = int(roi.top_left[1] * height)
        roi_x2 = int(roi.bottom_right[0] * width)
        roi_y2 = int(roi.bottom_right[1] * height)

        # Draw a filled rectangle on `overlay` in the stand's color
        cv2.rectangle(
            overlay,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            roi.color_bgr,
            thickness=-1
        )

        # Label the ROI with its name
        cv2.putText(
            overlay,
            roi.name,
            (roi_x1, roi_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2
        )

    # Merge with the original `frame` using some alpha for transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 3) Analyze detections
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                if cls == 0:  # "person" in COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_box = (x1, y1, x2, y2)

                    # We'll check overlap with each ROI
                    rois_in = []
                    person_area = compute_area(person_box)

                    # Avoid division by zero if box is degenerate
                    if person_area <= 0:
                        continue

                    for roi in rois:
                        rx1 = int(roi.top_left[0] * width)
                        ry1 = int(roi.top_left[1] * height)
                        rx2 = int(roi.bottom_right[0] * width)
                        ry2 = int(roi.bottom_right[1] * height)
                        roi_box = (rx1, ry1, rx2, ry2)

                        inter_area = compute_intersection_area(person_box, roi_box)
                        overlap_ratio = inter_area / float(person_area)

                        if overlap_ratio >= OVERLAP_THRESHOLD:
                            rois_in.append(roi)

                    # Determine bounding box color
                    if len(rois_in) == 0:
                        # Person is not in any stand
                        box_color = (0, 255, 0)  # Green
                        outside_count += 1
                    elif len(rois_in) == 1:
                        # Exactly one ROI
                        box_color = rois_in[0].color_bgr
                        rois_in[0].frame_attention_count += 1
                        rois_in[0].total_attention_count += 1
                    else:
                        # In multiple stands => red bounding box
                        box_color = (0, 0, 255)
                        for roi_in_multiple in rois_in:
                            roi_in_multiple.frame_attention_count += 1
                            roi_in_multiple.total_attention_count += 1

                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 4) Show the per-frame counts for each stand
    y_offset = 30
    for roi in rois:
        text = f"{roi.name}: {roi.frame_attention_count} in this frame"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 20
    
    # Also display how many are outside
    outside_text = f"Outside: {outside_count} in this frame"
    cv2.putText(frame, outside_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Collect data for time-series
    timeline_row = {"frame": frame_index, "Outside": outside_count}
    for roi in rois:
        timeline_row[roi.name] = roi.frame_attention_count
    timeline_data.append(timeline_row)

    # Show the video with overlays
    cv2.imshow('CCTV Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------
# 5. Summarize Results
# ---------------------------
print("Attention analysis results:")
for roi in rois:
    print(f"{roi.name} received {roi.total_attention_count} total attention counts.")

# ---------------------------
# 6. Generate Smoothed Time-Series Graphs
# ---------------------------

frames = [row["frame"] for row in timeline_data]

# For each ROI + "Outside," we build a smoothed series and plot it
plt.figure(figsize=(10, 6))

# Plot "Outside"
raw_outside = [row["Outside"] for row in timeline_data]
smooth_outside = smooth_data(raw_outside, window_size=5)
plt.plot(frames, smooth_outside, label="Outside")

# Plot each ROI
for roi in rois:
    raw_counts = [row[roi.name] for row in timeline_data]
    smooth_counts = smooth_data(raw_counts, window_size=5)
    plt.plot(frames, smooth_counts, label=roi.name)

plt.title("Smoothed Number of People by Frame (5-frame moving average)")
plt.xlabel("Frame")
plt.ylabel("People Count")
plt.legend()
plt.show()

# Alternatively, a bar chart of total attention (ROI only):
plt.figure(figsize=(6,4))
roi_names = [r.name for r in rois]
roi_totals = [r.total_attention_count for r in rois]
plt.bar(roi_names, roi_totals, color='orange')
plt.xlabel("ROI (Stand)")
plt.ylabel("Total Attention Count")
plt.title("Total Attention Per ROI")
plt.show()
