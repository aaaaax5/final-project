import socket # 소켓 프로그래밍에 필요한 API를 제공하는 모듈
import struct # 바이트(bytes) 형식의 데이터 처리 모듈
import pickle # 객체의 직렬화 및 역직렬화 지원 모듈
import cv2 # OpenCV(실시간 이미지 프로세싱) 모듈

ip = '192.168.2.59'
port = 50002

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((ip, port))
    print("연결 성공")

    data_buffer = b""

    data_size = 4
    while True:
        while len(data_buffer) < data_size:
            data_buffer += client_socket.recv(4096)

        packed_data_size = data_buffer[:data_size]
        data_buffer = data_buffer[data_size:] 
        frame_size = struct.unpack(">L", packed_data_size)[0]

        while len(data_buffer) < frame_size:
            data_buffer += client_socket.recv(4096)

        frame_data = data_buffer[:frame_size]
        data_buffer = data_buffer[frame_size:]
        print("수신 프레임 크기 : {} bytes".format(frame_size))
        frame = pickle.loads(frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
# 소켓 닫기
client_socket.close()
print('연결 종료')
cv2.destroyAllWindows()