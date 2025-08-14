from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Tiny model for speed

cap = cv2.VideoCapture(0)  # Webcam
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    results = model(frame, stream=True)  # Process frame
    for r in results:
        cv2.imshow("Webcam", r.plot())  # Display
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()