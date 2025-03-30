import cv2
from ultralytics import YOLO

# Loading pretrained YOLO model (will e downloaded on firs run)
model = YOLO("model/yolov8n.pt", "v8")

# Set dimensions of video frames
frame_width = 1280
frame_height = 720

# Video source is MP4 file stored locally
cap = cv2.VideoCapture("downloads/videos/download.mp4")

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No video frame avalable")
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Do prediction on image, with confidence greater than 80%
    detect_params = model.predict(source=[frame], conf=0.8, save=False)

    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):

            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            c = box.cls
            # Name of object detected (e.g. 'bird')
            class_name = model.names[int(c)]

        if 'bird' in class_name.lower():

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                (0, 255, 0),
            )

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(
                frame,
                class_name + " " +str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # End program when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
