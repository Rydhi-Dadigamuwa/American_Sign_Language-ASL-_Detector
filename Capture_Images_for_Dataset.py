import cv2
import os

# Define classes to include all letters of the alphabet (ASCII values for A-Z)
class_names = [chr(i) for i in range(65, 91)]

# YOLOv8 input size
target_size = (640, 640)


cap = cv2.VideoCapture(0)
base_dir = "images"

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for class_name in class_names:
    class_path = os.path.join(base_dir, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

image_id = 0  # counter to save images with unique names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # Save image if a class key is pressed (for letters A-Z)
    if 65 <= key <= 90 or 97 <= key <= 122:
        if 97 <= key <= 122:  # Convert lowercase to uppercase
            key -= 32
        class_index = key - 65  # Convert key press to class index
        class_name = class_names[class_index]

        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        image_path = os.path.join(base_dir, class_name, f'{class_name}_{image_id}.jpg')
        cv2.imwrite(image_path, resized_frame)
        print(f'Image saved to {image_path}')
        image_id += 1

    if key == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()