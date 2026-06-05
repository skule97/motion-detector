#%%
import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"video{i}: {'OK' if ret else 'kein Frame'}")
        cap.release()
    else:
        print(f"video{i}: nicht öffenbar")