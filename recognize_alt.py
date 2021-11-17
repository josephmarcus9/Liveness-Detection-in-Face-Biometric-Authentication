import itertools
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import SVC
import cv2
import os
from sklearn.model_selection import GridSearchCV
import numpy as np
from metrics import metric
from sklearn.preprocessing import LabelEncoder


def file_traverse(path, store):
    desc = LocalBinaryPatterns(8, 1)
    data = []
    labels = []
    file = open(path)

    for line in itertools.islice(file, 20):
        line = line.replace("\\", "/")
        image_path = real_train + '/' + line
        path_1 = image_path.strip().split('/')
        retain = path_1[:6]
        retain.append(store)
        rest = path_1[7:]
        new_file = os.path.join('/', *(retain + rest))
        path_2, file = os.path.split(new_file)
        x = os.path.join(path_2, file)
        image = cv2.imread(x)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        labels.append(retain[-1])
        data.append(hist)
    return data, labels


def svm_model(train_data, train_labels, params):
    model1 = SVC(**params, probability=True)  # ** unpacks dict
    model1.fit(train_data, train_labels)
    return model1


def predict(model, data):
    pridicted_label_set = []
    for hist in data:
        hist_new = np.reshape(hist, (1, len(hist)))
        prediction = model.predict(hist_new)
        pridicted_label_set.append(prediction[0])
    return pridicted_label_set


if __name__ == "__main__":
    real_train = '/Users/josephmarcus/Desktop/Datasets/NUAA_cropped_normalised/client_test_raw.txt'
    sub_dir = 'ClientRaw'
    train_real_data, train_real_labels = file_traverse(real_train, sub_dir)
    print(sub_dir, ':', train_real_labels.count('ClientRaw'))  # 200 live

    spoof_train = '/Users/josephmarcus/Desktop/Datasets/NUAA_cropped_normalised/imposter_test_raw.txt'
    sub_dir_1 = 'ImposterRaw'
    train_spoof_data, train_spoof_labels = file_traverse(spoof_train, sub_dir_1)
    print(sub_dir_1, ':', train_spoof_labels.count('ImposterRaw'))  # 200 live

    train_data = train_real_data + train_spoof_data
    train_labels = train_real_labels + train_spoof_labels

    params_0 = {'random_state': 42}
    model1 = svm_model(train_data, train_labels, params_0)
    params = {'C': [250, 255, 260, 270], 'kernel': ['linear', 'rbf'], 'gamma': [160, 170, 180, 200, 150]}
    grid = GridSearchCV(model1, param_grid=params, cv=10)
    grid.fit(train_data, train_labels)
    best = grid.best_params_
    print(best)

    params_2 = {'C': best['C'], 'random_state': 42, 'kernel': best['kernel'], 'gamma': best['gamma']}
    model = svm_model(train_data, train_labels, params_2)

    real_test = '/Users/josephmarcus/Desktop/Datasets/NUAA_cropped_normalised/client_train_raw.txt'
    test_real_data, test_real_labels = file_traverse(real_test, sub_dir)
    print(sub_dir, ':', test_real_labels.count('ClientRaw'))  # 200 live

    spoof_test = '/Users/josephmarcus/Desktop/Datasets/NUAA_cropped_normalised/imposter_train_raw.txt'
    test_spoof_data, test_spoof_labels = file_traverse(spoof_test, sub_dir_1)
    print(sub_dir_1, ':', test_spoof_labels.count('ImposterRaw'))  # 200 live

    test_data = test_real_data + test_spoof_data
    test_labels = test_real_labels + test_spoof_labels
    predict_proba = model.predict_proba(test_data)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(test_labels)
    apcer, bpcer, hter, eer = metric(predict_proba, integer_encoded)

    print('APCER:', apcer)
    print('BPCER:', bpcer)
    print('HTER:', hter)
    print('EER', eer)
