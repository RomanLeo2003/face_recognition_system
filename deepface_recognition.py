import os
from deepface import DeepFace
from transliterate import translit
import cv2
from PIL import Image
from time import sleep
import pickle

def xywh2xyxy(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y - h
    return x1, y1, x2, y2

class DeepFaceHandler:
    def __init__(self, faces_path: str, detector: str = 'mediapipe', extractor: str = 'VGG-Face'):
        self.faces_path = faces_path
        self.detector = detector
        self.extractor = extractor

    def form_path(self, name_employee: str):
        eng_name_employee = translit(name_employee, language_code='ru', reversed=True)
        eng_name_employee = '_'.join(eng_name_employee.split())
        new_path = self.faces_path + '/' + eng_name_employee + '.jpg'
        return new_path

    def save_face(self, path_to_image: str, name_employee: str):
        '''
        :param path: path to faces folder
        :param path_to_image: full path to new image
        :param name_employee: name of new employee in format: Имя Фамилия
        :return:
        '''
        img = Image.open(path_to_image)
        new_path = self.form_path(name_employee)
        img.save(new_path)
        embs = []
        try:
            with open(self.faces_path + r"\representations_vgg_face.pkl", 'rb') as file:
                embs = pickle.load(file)
        except:
            pass

        new_face_embedding = DeepFace.represent(img_path=path_to_image, model_name=self.extractor, detector_backend=self.detector)
        if [new_path, new_face_embedding[0]['embedding']] not in embs:
            embs.append([new_path, new_face_embedding[0]['embedding']])

        with open(self.faces_path + r"\representations_vgg_face.pkl", 'wb') as file:
            pickle.dump(embs, file)


    def delete_face(self, name_employee):
        path_to_delete = self.form_path(name_employee)
        if path_to_delet[path_to_delet.rfind('/') + 1:] in os.listdir(self.faces_path):
            os.remove(path_to_delet)
            with open(self.faces_path + r"\representations_vgg_face.pkl", 'rb') as file:
                embs = pickle.load(file)
                for emb in embs:
                    if emb[0] == path_to_delete:
                        embs.remove(emb)
            print('Данные сотрудника удалены')
        else:
            print('Сотрудника нет в базе лиц')


    def main_loop(self):
        cap = cv2.VideoCapture(0)
        first_employee_photo = self.faces_path + "/" + os.listdir(self.faces_path)[1]
        DeepFace.find(first_employee_photo, detector_backend=self.detector, model_name=self.extractor, db_path=self.faces_path)
        while True:
            ret, frame = cap.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            cv2.imshow('image', frame)
            try:
                d = DeepFace.extract_faces(frame, detector_backend=self.detector, align=True, enforce_detection=False)
                print(d[0]['confidence'])
                df = DeepFace.find(frame, detector_backend=self.detector, model_name=self.extractor, db_path=self.faces_path,
                                   enforce_detection=False)
                # print(df[0])
                face_path = df[0]['identity'][0]
                face_name = translit(face_path[face_path.rfind("/") + 1:-4], language_code="ru").split("_")
                # for situations like this: Роман леонтЬев
                face_name = " ".join([fn[0].upper() + fn[1:].lower() for fn in face_name])
                print(f'Распознан {face_name}')
            except:
                print('Лицо не распознано')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

path = r"C:\Users\user\Downloads\faces"
dfh = DeepFaceHandler(path, detector='mediapipe')
dfh.save_face(r"C:\Users\user\Downloads\Roman_Leontev.jpg", 'Роман Леонтьев')
dfh.main_loop()