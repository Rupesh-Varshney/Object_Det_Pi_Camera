import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv8 model (n = nano, s = small, m = medium)
    # The nano version is fastest for real-time webcam use
    model = YOLO("yolov8n.pt") 

    # 2. Initialize the laptop camera
    # '0' is usually the built-in webcam
    video="http://10.77.67.61:5000"
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        
        if not success:
            break

        # 3. Run YOLOv8 inference on the frame
        # stream=True is more memory-efficient for continuous video
        results = model(frame, stream=True)

        # 4. Visualize the results on the frame
        for r in results:
            # This returns the original image with boxes/labels drawn on it
            annotated_frame = r.plot()

        # Display the resulting frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # 5. Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()