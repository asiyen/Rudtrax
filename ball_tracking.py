import cv2
import numpy as np
import time

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the range of colors for the white ball in HSV
lower_color = np.array([0, 0, 200])
upper_color = np.array([255, 255, 255])

# Initialize the previous center of the contour and timestamps
prev_center = None
prev_time = None
prev_prev_center = None
prev_prev_time = None

# Loop through frames
while True:
    # Read the current frame
    ret, frame = cap.read()
    current_time = time.time()
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
        if prev_prev_center is not None and prev_prev_time is not None:
            dt = current_time - prev_prev_time
            if dt > 0.5:
                dx = cX - prev_prev_center[0]
                dy = cY - prev_prev_center[1]
                v = (dx**2 + dy**2)**0.5 / dt
                print(f'Velocity: {v:.2f} pixels/s')
                prev_prev_center = None
                prev_prev_time = None
            else:
                prev_prev_center = prev_center
                prev_prev_time = prev_time
        else:
            prev_prev_center = prev_center
            prev_prev_time = prev_time
        prev_center = (cX, cY)
        prev_time = current_time

    # Show the frame
    cv2.imshow("Tracking", frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
