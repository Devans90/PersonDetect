import cv2
from typing import List, Tuple
from pydantic import BaseModel
from ultralytics import YOLO

# Define a data model for regions of interest (ROIs)
class ROI(BaseModel):
    name: str
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    attention_count: int = 0

# Initialize ROIs
rois = [
    ROI(name="Stand 1", top_left=(50, 50), bottom_right=(200, 200)),
    ROI(name="Stand 2", top_left=(250, 50), bottom_right=(400, 200)),
    # Add more stands as needed
]

# Load the YOLOv11 model
model = YOLO('yolo11l.pt')  # 'n' denotes the nano version; choose according to your needs

# Initialize video capture
cap = cv2.VideoCapture('cctv_footage.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection
    results = model(frame)

    # Iterate over each result (one per image/frame)
    for result in results:
        # Check if 'boxes' attribute exists
        if hasattr(result, 'boxes') and result.boxes is not None:
            # Iterate over detected boxes
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                if cls == 0:  # Class 0 is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Check if person is in any ROI
                    for roi in rois:
                        if (roi.top_left[0] <= person_center[0] <= roi.bottom_right[0] and
                                roi.top_left[1] <= person_center[1] <= roi.bottom_right[1]):
                            roi.attention_count += 1
                            # Draw rectangle around detected person
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Label the ROI
                            cv2.putText(frame, roi.name, roi.top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('CCTV Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Output attention analysis results
for roi in rois:
    print(f"{roi.name} received {roi.attention_count} attention counts.")
