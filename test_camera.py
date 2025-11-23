import cv2

def main():
    # 0 = default camera (Mac laptop camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera. Check permissions in System Settings > Privacy & Security > Camera.")
        return

    print("✅ Camera opened. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Show the frame in a window
        cv2.imshow("Cano - Camera Test", frame)

        # Wait for 1ms, quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()