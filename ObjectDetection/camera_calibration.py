import cv2
import numpy as np
import glob

# 체스보드 크기
chessboard_size = (7, 5)

# 체스보드의 3D 점 준비
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 3D 점과 2D 점 저장 리스트
objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 캘리브레이션 이미지 경로
images = glob.glob('/home/aa/dev_ws/final_project/aruco/checkerboard_pictures/robot_pictures/*.jpg')

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        # 코너 그리기 및 보여주기
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    
    else:
        print(f"체스보드 코너를 찾지 못했습니다")

cv2.destroyAllWindows()

# 카메라 캘리브레이션
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# 캘리브레이션 결과 저장
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)