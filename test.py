import cv2

fname = 'C:\\Users\\tkdwl\\py\\Opencv\\image\\test.jpg'
img = cv2.imread(fname)
"""original = cv2.imread(fname, cv2.IMREAD_COLOR)
gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
unchange = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

cv2.imshow('Original', original)
cv2.imshow('Gray', gray)
cv2.imshow('Unchange', unchange)"""
"""yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('H channel', yuv_img[:, :, 0])
cv2.imshow('S channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])"""
import numpy as np
#영상이동
"""num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols+70,num_rows+110))
cv2.imshow('Translation', img_translation)"""
#영상회전
"""num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('rotation',img_rotation)"""
#영상스케일링
"""img_scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled = cv2.resize(img, (500,450), interpolation=cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)"""
#어파인 변환
"""rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0],[0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(rows-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
cv2.imshow("input", img)
cv2.imshow('output', img_output)"""
#좌우반전
"""capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = capture.read()    
    rows, cols = frame.shape[:2]
    src_points = np.float32([[0,0], [cols-1,0],[0,rows-1]])
    dst_points = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(frame, affine_matrix, (cols,rows))
    cv2.imshow("original", frame)
    cv2.imshow("Affine Transtion", img_output)   
    if cv2.waitKey(1) == ord('q'):  
            break

capture.release()"""    
#영상와핑
"""import math
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = capture.read()    
    rows, cols = frame.shape[:2]
    #both horizontal and vertical
    img_output = np.zeros(frame.shape, dtype = frame.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(20.0*math.sin(2*3.14*i/150))
            offset_y = int(20.0*math.cos(2*3.14*j/150))
            if i+offset_y < rows and j+offset_x < cols:
                img_output[i,j] = frame[(i+offset_y)%rows, (j+offset_x)%cols]
            else:
                img_output[i,j]=0 
    cv2.imshow("Multidirectional wave", img_output)
    #concave effect
    img_output = np.zeros(frame.shape, dtype = frame.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(128.0*math.sin(2*3.14*i/(2*cols)))
            offset_y = 0
            if i+offset_y < rows and j+offset_x < cols:
                img_output[i,j] = frame[i, (j+offset_x)%cols]
            else:
                img_output[i,j]=0 
    cv2.imshow("Concave", img_output)
    if cv2.waitKey(1) == ord('q'):  
            break

capture.release()"""
cv2.waitKey(0)
cv2.destroyAllWindows()