import cv2

# Start webcam
cap = cv2.VideoCapture(0)

# 💡 Add this line right before imshow:
cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show webcam feed
    cv2.imshow("Webcam Test", frame)  # ← This is where we added the window setup above

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
