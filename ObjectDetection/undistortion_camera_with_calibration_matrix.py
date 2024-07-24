import cv2
import numpy as np
import glob

# 카메라 행렬과 왜곡 계수
camera_matrix = np.array([[501.71184364, 0, 319.93415777],
                          [0, 500.97002316, 245.20985799],
                          [0, 0, 1]])

dist_coeffs = np.array([0.20046918, -0.5471712, -0.00194516, -0.00144732, 0.42454179])


# 이미지를 로드하고 왜곡을 보정합니다ㅂ
images = glob.glob('/home/aa/Downloads/got_test_iamge/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 왜곡 보정
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 결과 표시
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()