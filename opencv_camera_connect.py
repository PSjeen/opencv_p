import argparse
import cv2
import numpy as np
"""# opencv python 코딩 기본 틀
# 카메라 영상을 받아올 객체 선언 및 설정(영상 소스, 해상도 설정)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 무한루프
while True:
    ret, frame = capture.read()     # 카메라로부터 현재 영상을 받아 frame에 저장, 잘 받았다면 ret가 참
    cv2.imshow("original", frame)   # frame(카메라 영상)을 original 이라는 창에 띄워줌 
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프가 멈춤
            break

capture.release()                   # 캡처 객체를 없애줌
cv2.destroyAllWindows()             # 모든 영상 창을 닫아줌"""

"""cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
"""
#키보드입력
"""def argument_parser():
    parser = argparse.ArgumentParser(description="Change color space of the input video stream using keyboard controls. GrayScale - 'g', YUV - 'y', HSV - 'h'")
    return parser
if __name__=='__main__':
    args = argument_parser().parse_args()
    cap = cv2.VideoCapture(0)
    cur_char = -1
    prev_char = -1
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        c = cv2.waitKey(1)
        if c == 27:
            break

        if c >-1 and c != prev_char:
            cur_char = c
        prev_char = c
        if cur_char == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_char == ord('y'):
            output = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
        elif cur_char == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            output = frame
        cv2.imshow('', output)"""
#마우스 입력
def detect_quadrant(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x> width/2:
            if y > height/2:
                point_top_left = (int(width/2), int(height/2))
                point_bottom_right = (width-1, height-1)
            else:
                point_top_left = (int(width/2), 0)
                point_bottom_right = (width-1, int(height/2))
        else:
            if y > height/2:
                point_top_left = (0, int(height/2))
                point_bottom_right = (int(width/2), height-1)
            else:
                point_top_left = (0, 0)
                point_bottom_right = (int(width/2), int(height/2))
        cv2.rectangle(img, (0,0), (width-1,height-1), (255,255,255), -1)
        cv2.rectangle(img, point_top_left, point_bottom_right,(0,100,0), -1)
if __name__=='__main__':
    width, height = 640, 480
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    cv2.namedWindow('input window')
    cv2.setMouseCallback('input window', detect_quadrant)
    while True:
        cv2.imshow('input window', img)
        c =cv2.waitKey(10)
        if c == 27:
            break

#cap.release()
cv2.destroyAllWindows()
        