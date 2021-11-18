import cv2
import os
from numpy.random import permutation
from imutils import paths

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def crop_face(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=6)
    if len(faces) == 0:
        return image
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            return roi_color


def process_db(db_dir,cropped_word):
    for imagePath in permutation(list(paths.list_images(db_dir))):
        image = cv2.imread(imagePath)
        crop = crop_face(image)
        path_1 = imagePath.split("/")
        retain = path_1[:5]

        retain.append(cropped_word)
        rest = path_1[6:]
        new_file = os.path.join('/', *(retain + rest))
        path_2, file = os.path.split(new_file)
        if not os.path.exists(path_2):
            os.makedirs(path_2)
        x = os.path.join(path_2, file)
        img_resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        norm_img = cv2.normalize(img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(x, norm_img)


if __name__ == "__main__":
    db_dir_1 = '/Users/josephmarcus/Desktop/Datasets/NUAA/ImposterRaw'
    cropped_word_1 = 'NUAA_cropped'
    process_db(db_dir_1, cropped_word_1)
