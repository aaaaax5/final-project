import socket # 소켓 프로그래밍에 필요한 API를 제공하는 모듈
import struct # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle # 객체의 직렬화 및 역직렬화 지원 모듈
import cv2 # OpenCV(실시간 이미지 프로세싱) 모듈

ip = '192.168.2.59'
port = 50002

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server_socket.bind((ip, port))

server_socket.listen(10) 
print('클라이언트 연결 대기')

client_socket, address = server_socket.accept()
print('클라이언트 ip 주소 :', address[0])

capture = cv2.VideoCapture(0)
# 프레임 크기 지정
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로

print("연결 성공")
# 메시지 수신
while True:

    retval, frame = capture.read()

    retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    frame = pickle.dumps(frame)
    print("전송 프레임 크기 : {} bytes".format(len(frame)))

    client_socket.sendall(struct.pack(">L", len(frame)) + frame)

capture.release()

client_socket.close()
server_socket.close()
print('연결 종료')
cv2.destroyAllWindows()