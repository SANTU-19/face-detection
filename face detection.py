#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

# Load the pre-trained Haar cascade face detector
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
video_cap = cv2.VideoCapture(0)

# Function to handle mouse clicks
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 50:
            print("Exit button clicked")
            global exit_program
            exit_program = True

# Create a window and bind the mouse callback function
cv2.namedWindow("face_detect")
cv2.setMouseCallback("face_detect", mouse_click)

exit_program = False

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Face count message
    face_count = len(faces)
    if face_count==1:
        msg=f'{face_count} face is detected'
    elif face_count>1:
        msg=f'{face_count} face are detected'
    else:
        msg = "No face detected"

    # Draw face count message
    cv2.putText(video_data, msg, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Draw exit button
    cv2.rectangle(video_data, (10, 10), (110, 50), (0, 0, 255), -1)
    cv2.putText(video_data, "EXIT", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("face_detect", video_data)

    # Keyboard exit or mouse click exit
    if cv2.waitKey(10) == ord("a") or exit_program:
        break

# Cleanup
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




