
# Python 2/3 compatibility
#from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import RPi.GPIO as GPIO
import pyfirmata

# local modules
from video import create_capture
from common import clock, draw_str

board = pyfirmata.Arduino('/dev/ttyUSB0') #아두이노에 연결합니다.

led_builtin = board.get_pin('d:13:o') # 디지털(digital) 핀(pin) 13번을 출력(output) 모드로 가져옵니다.
servo_pin = board.get_pin('d:9:p')

GPIO.setmode(GPIO.BCM)

detect = 0
end_time = 0
count = 0
n=0
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        n=1
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))
    
    while True:
        global count
        global servo_pin
        n=0
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
                n=1
                start_time = time.time()
        if n==0:
            end_time = time.time()
            led_builtin.write(0)
            if count == 1:
                    for high_time in range (99, 1, -1):
                        servo_pin.write(high_time/100.0) # for 반복문에 실수가 올 수 없으므로 /10.0 로 처리함. 
                        time.sleep(0.02)
                    count = 0
        else:
            print(f"time : {int(round((start_time - end_time) * 1000))}ms")
            if int(round((start_time - end_time) * 1000)) > 3000:
                dectect = 1
                led_builtin.write(1)
                if count == 0:
                    for high_time in range (1, 100):
                        servo_pin.write(high_time/100.0) # for 반복문에 실수가 올 수 없으므로 /10.0 로 처리함. 
                        time.sleep(0.02)
                    count = 1
            else:
                dectect = 0
                led_builtin.write(0)
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)
        
        """
        if detect == 1:
            led_builtin.write(1)
        else :
            led_builtin.write(0)
            """
        
        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
pwm.stop()
GPIO.cleanup()
