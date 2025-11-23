import cv2

def main():
    print("🔍 Scanning for available camera devices...\n")

    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera index {i} is AVAILABLE")
            cap.release()
        else:
            print(f"❌ Camera index {i} is NOT available")

if __name__ == "__main__":
    main()