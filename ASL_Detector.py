import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO('C:/Users/DELL/PycharmProjects/American_Sign_Language_Letters_Detector/Model/best.pt')

threshold = 0.5

ret, frame = cap.read()
H, W, _ = frame.shape

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()