import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the range of colors for the white ball in HSV
lower_color = np.array([0, 0, 200])
upper_color = np.array([255, 255, 255])

# Initialize the previous frame and the previous center of the contour
prev_frame = None
prev_center = None

# Loop through frames
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to only select white colors
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Find contours in the frame
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw a circle at the center of the contour
        cv2.circle(frame, (cX, cY), 10, (255, 0, 0), -1)

        # Calculate the velocity of the ball if the previous frame and previous center are available
        if prev_frame is not None and prev_center is not None:
            dt = 1/30 # assuming that the video is 30 FPS
            dx = cX - prev_center[0]
            dy = cY - prev_center[1]
            vx = dx / dt
            vy = dy / dt
            v = (vx**2 + vy**2)**0.5
            print(f'Velocity: {v:.2f} pixels/s')

        # Update the previous frame and previous center
        prev_frame = frame
        prev_center = (cX, cY)

    # Show the frame
    cv2.imshow("Tracking", frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
