
from bluetooth import *

socket = BluetoothSocket( RFCOMM )
socket.connect(("00:18:E4:34:D1:C9", 1))
print("bluetooth connected!")


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
            msg = 'B'
            socket.send(msg)
        else:
            print(f"time : {int(round((start_time - end_time) * 1000))}ms")
            if int(round((start_time - end_time) * 1000)) > 3000:
                dectect = 1
                msg = 'A'
                socket.send(msg)
            else:
                dectect = 0
        cv.imshow('facedetect', vis)
        
        """
        if detect == 1:
            led_builtin.write(1)
        else :
            led_builtin.write(0)
            """
        
        if cv.waitKey(5) == 27:
            print("finished")
            socket.close()
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()