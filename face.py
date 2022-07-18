import cv2
import time
face_cascade = cv2.CascadeClassifier('C:\\Program Files\\WindowsApps\\www.cyberlink.com.photodirectorforlge_8.0.3022.0_x64__srrwvbh8chymt\\PhotoDirector8\\haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
end_time=0
dectect = 0
while True:
    n=0
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation= cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        n=1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        start_time = time.time()
    if n==0:
        end_time = time.time()
    else:
        print(f"time : {int(round((start_time - end_time) * 1000))}ms")
        if int(round((start_time - end_time) * 1000)) > 3000:
            dectect = 1
        else:
            dectect = 0
    
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()