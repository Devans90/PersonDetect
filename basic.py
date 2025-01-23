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
# Example: top_left=(0.1, 0.1) means 10% from the left, 10% from the top, etc.
# Assign unique BGR colors to each stand so you can differentiate them visually.
rois = [
    ROI(name="Stand 1",
        top_left=(0.05, 0.05),
        bottom_right=(0.30, 0.80),
        color_bgr=(255, 200, 100)),  # Light blue
    ROI(name="Stand 2",
        top_left=(0.3, 0.05),
        bottom_right=(0.80, 0.60),
        color_bgr=(100, 200, 255)),  # Light orange-ish
    # Add more stands as needed, with unique colors
]


# ---------------------------
# 2. Initialize Model
# ---------------------------

model = YOLO('yolo11n.pt')  
# You can play around with confidence threshold, iou threshold, etc.:
# results = model(frame, conf=0.25, iou=0.45)

# ---------------------------
# 3. Video Capture
# ---------------------------

cap = cv2.VideoCapture('cctv_footage.mp4')

# (Optional) Store historical data for time-series plotting:
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

    # -------------------------------------------
    # 3A. Predict with YOLO (person detection)
    # -------------------------------------------
    results = model(frame)

    # Prepare an overlay copy for semi-transparent ROI rectangles
    overlay = frame.copy()

    # -------------------------------------------
    # 3B. Draw semi-transparent overlays for each stand ROI
    # -------------------------------------------
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
            (255, 255, 255),  # Text color (white)
            2
        )

    # Merge with the original `frame` using some alpha for transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # -------------------------------------------
    # 3C. Check each detection and draw bounding box
    # -------------------------------------------
    # We'll store the bounding box color logic:
    # - Green if in no ROI
    # - ROI color if in exactly one ROI
    # - Red if in multiple ROIs
    results = model(frame)
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                if cls == 0:  # Class 0 is 'person' in COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Determine which ROIs this person is in
                    rois_in = []
                    for roi in rois:
                        roi_x1 = int(roi.top_left[0] * width)
                        roi_y1 = int(roi.top_left[1] * height)
                        roi_x2 = int(roi.bottom_right[0] * width)
                        roi_y2 = int(roi.bottom_right[1] * height)

                        if (roi_x1 <= person_center[0] <= roi_x2 and
                                roi_y1 <= person_center[1] <= roi_y2):
                            rois_in.append(roi)

                    # Choose bounding box color based on how many stands the person is in
                    if len(rois_in) == 0:
                        # Not in any stand
                        box_color = (0, 255, 0)  # Green
                    elif len(rois_in) == 1:
                        # In exactly one ROI => use that ROI's color
                        box_color = rois_in[0].color_bgr
                        # Increment the ROI's counters
                        rois_in[0].frame_attention_count += 1
                        rois_in[0].total_attention_count += 1
                    else:
                        # In multiple ROIs => red bounding box
                        box_color = (0, 0, 255)
                        # If you want to increment counters for all, you can do so:
                        for r in rois_in:
                            r.frame_attention_count += 1
                            r.total_attention_count += 1

                    # Draw the bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

    # -----------------------------------------------------
    # 3D. (Optional) Show the per-frame ROI counts
    # -----------------------------------------------------
    y_offset = 30
    for roi in rois:
        text = f"{roi.name}: {roi.frame_attention_count} in this frame"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        y_offset += 20

    # Collect timeline data (for each ROI, how many were detected)
    timeline_row = {roi.name: roi.frame_attention_count for roi in rois}
    timeline_row["frame"] = frame_index
    timeline_data.append(timeline_row)

    # Show the video with overlays
    cv2.imshow('CCTV Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------
# 4. Summarize Results
# ---------------------------
print("Attention analysis results:")
for roi in rois:
    print(f"{roi.name} received {roi.total_attention_count} total attention counts.")

# ---------------------------
# 5. Generate Graphs
# ---------------------------
# Let's create a simple time-series of how many persons per frame for each ROI.

frames = [row["frame"] for row in timeline_data]

plt.figure(figsize=(10, 6))
for roi in rois:
    counts_per_frame = [row[roi.name] for row in timeline_data]
    plt.plot(frames, counts_per_frame, label=roi.name)

plt.title("Number of People in Each ROI by Frame")
plt.xlabel("Frame")
plt.ylabel("People Count")
plt.legend()
plt.show()

# Alternatively, you might want a simple bar chart of total attention:
plt.figure(figsize=(6,4))
roi_names = [r.name for r in rois]
roi_totals = [r.total_attention_count for r in rois]
plt.bar(roi_names, roi_totals, color='orange')
plt.xlabel("ROI (Stand)")
plt.ylabel("Total Attention Count")
plt.title("Total Attention Per ROI")
plt.show()