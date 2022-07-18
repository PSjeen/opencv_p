import cv2
import time

face_cascade = cv2.CascadeClassifier('C:\\Program Files\\WindowsApps\\www.cyberlink.com.photodirectorforlge_8.0.3022.0_x64__srrwvbh8chymt\\PhotoDirector8\\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\tkdwl\\py\\cv_env\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
end_time = 0
while True:
    n=0
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print("Number of faces dectected: ", str(len(face_rects)))
    for (x,y,w,h) in face_rects:
        face_img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey ,ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
            n=1
            start_time = time.perf_counter()
    if n==0:
        end_time = time.perf_counter()
    else:
        print(f"time : {int(round((start_time - end_time) * 1000))}ms")
    cv2.imshow('Face Detector', face_img)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()