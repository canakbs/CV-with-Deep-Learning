from ultralytics import YOLO
import cv2

model=YOLO("traffic-sign-model/weights/last.pt")

image_path="test1.jpg"
image=cv2.imread(image_path)

results=model(image_path)

results.show()

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    label=(f"Box: ({x1}, {y1}), ({x2}, {y2}), Confidence: {conf}, Class: {cls}")

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
