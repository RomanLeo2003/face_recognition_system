from deepface import DeepFace
import cv2
from PIL import Image
faces_path = r"C:\Users\user\Downloads\faces"
first_employee_photo = faces_path + fr'\roman_leontev_{0}.jpg'
DeepFace.find(first_employee_photo, detector_backend='mediapipe', model_name='VGG-Face', db_path=faces_path)


def main_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('image', frame)
        try:
            df = DeepFace.find(frame, detector_backend='retinaface', model_name='VGG-Face', db_path=faces_path)
            print(df[0])
            for face_path in df[0]['identity']:
                if face_path[-5] == '0':
                    print(f'Распознан {face_path}')
                    break
        except:
            print('Лицо не распознано')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main_loop()