# import cv2 as cv

# dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
# parameters =  cv.aruco.DetectorParameters()
# detector = cv.aruco.ArucoDetector(dictionary, parameters)

# frame = cv.imread('DICT_4X4_100_id_23.png')

# markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl


vid = cv2.VideoCapture(0)

vid.set(3, 640)
vid.set(4, 480)

camera_matrix = np.array([[501.71184364, 0, 319.93415777],
                          [0, 500.97002316, 245.20985799],
                          [0, 0, 1]])

dist_coeffs = np.array([0.20046918, -0.5471712, -0.00194516, -0.00144732, 0.42454179])



marker_size = 30
marker_3d_edges = np.array([    [0,0,0],
                                [0,marker_size,0],
                                [marker_size,marker_size,0],
                                [marker_size,0,0]], dtype = 'float32').reshape((4,1,3))


while (True):

    ret, frame = vid.read()
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    if(ret):
        # 마커(marker) 검출
        corners, ids, rejectedCandidates = detector.detectMarkers(frame)

        # 검출된 마커들의 꼭지점을 이미지에 그려 확인
        for corner in corners:
            corner = np.array(corner).reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner

            topRightPoint    = (int(topRight[0]),      int(topRight[1]))
            topLeftPoint     = (int(topLeft[0]),       int(topLeft[1]))
            bottomRightPoint = (int(bottomRight[0]),   int(bottomRight[1]))
            bottomLeftPoint  = (int(bottomLeft[0]),    int(bottomLeft[1]))

            # PnP
            ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, camera_matrix, dist_coeffs)
            if(ret):
                x=round(tvec[0][0],0);
                y=round(tvec[1][0],0);
                z=round(tvec[2][0],0);
                rx=round(np.rad2deg(rvec[0][0]),0);
                ry=round(np.rad2deg(rvec[1][0]),0);
                rz=round(np.rad2deg(rvec[2][0]),0);
                # PnP 결과를 이미지에 그려 확인
                text1 = f"{x},{y},{z}"
                text2 = f"{rx},{ry},{rz}"
                cv2.putText(frame, text1, (int(topLeft[0]-10),   int(topLeft[1]+10)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255))
                cv2.putText(frame, text2, (int(topLeft[0]-10),   int(topLeft[1]+40)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255))
        

    cv2.imshow('Detected ArUco markers', frame)

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

