import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

while True:
    ret, frame = capture.read(cv2.IMREAD_GRAYSCALE)
    rows, cols = frame.shape[:2]
    #블러링
    """kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
    kernel_3x3 = np.ones((3,3), np.float32) / 9.0
    kernel_5x5 = np.ones((5,5), np.float32) / 25.0
    output = cv2.filter2D(frame, -1, kernel_identity)
    cv2.imshow('id filter', output)
    output = cv2.filter2D(frame, -1, kernel_3x3)
    cv2.imshow('3x3 filter', output)
    output = cv2.filter2D(frame, -1, kernel_5x5)
    cv2.imshow('5x5 filter', output) """   
    #에지탐지
    """sobel_horizontal = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imshow('Sobel horizontal', sobel_horizontal)
    cv2.imshow('Sobel vertical', sobel_vertical)
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    cv2.imshow("Laplacian", laplacian)
    canny = cv2.Canny(frame, 50, 240)
    cv2.imshow("cannt", canny)"""
    #모션블러
    """size =15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(frame, -1, kernel_motion_blur)
    cv2.imshow("Motion Blur", output)"""
    #샤프닝
    """kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    kernel_sharpen_2 = np.array([[1,1,1],[1,7,1],[1,1,1]])
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]]) / 8.0
    output1 = cv2.filter2D(frame, -1, kernel_sharpen_1)
    output2 = cv2.filter2D(frame, -1, kernel_sharpen_2)
    output3 = cv2.filter2D(frame, -1, kernel_sharpen_3)
    cv2.imshow("Sharpening", output1)
    cv2.imshow("Excessive Sharpening", output2)
    cv2.imshow("Edge Enhancement", output3)"""
    #엠보싱
    """kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    kernel_emboss_2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
    kernel_emboss_3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    #converting the image to grayscale
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    output1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128
    output2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128
    output3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128
    cv2.imshow("Embossing-south west", output1)
    cv2.imshow("Embossing-south east", output2)
    cv2.imshow("Embossing-north west", output3)"""
    #침식 팽창
    """kernel = np.ones((5,5), np.uint8)
    img_erosion = cv2.erode(frame, kernel, iterations=1)
    img_dilation = cv2.dilate(frame, kernel, iterations=1)
    cv2.imshow("Erosion", img_erosion)
    cv2.imshow("Dilation", img_dilation)"""
    #비네트
    """#generating vignette mask using gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, 200) #(int(1.5*cols))
    kernel_y = cv2.getGaussianKernel(rows, 200) #(int(1.5*rows))
    kernel = kernel_y * kernel_x.T
    mask = 255*kernel/np.linalg.norm(kernel)        # + mask = mask[int(0.5*rows):, ing(0.5*cols):]
    output = np.copy(frame)
    #applying the mask to each channel in the input img
    for i in range(3):
        output[:,:,i] = output[:,:,i] *mask
    cv2.imshow("Vignette", output)"""
    #영상대비
    """frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #그레이 영상만 흐스토그램 균일화 가능
    histeq = cv2.equalizeHist(frame_gray)
    cv2.imshow("Histogram equalized", histeq)
    #컬러영상 히스토그램 균일화
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
    frame_output = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow("color Histogram equalized", frame_output)"""
    cv2.imshow("original", frame)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows
