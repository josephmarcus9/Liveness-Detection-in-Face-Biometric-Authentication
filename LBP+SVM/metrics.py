from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def metric(predict_proba, labels2):
    predict = np.argmax(predict_proba, axis=1)
    labels2 = np.asarray(labels2)
    con_matrix = confusion_matrix(labels2, predict)
    TP = con_matrix[0][0]
    FN = con_matrix[0][1]
    FP = con_matrix[1][0]
    TN = con_matrix[1][1]

    print("True Positive-->The classifier model predicted " + str(
        TP) + " real(positive) samples as real(Positive)")  # reals correctly predicted
    print("False Negative-->The classifier model predicted " + str(
        FN) + " real(Positive) samples as spoof(Negative)")  # real predicted as spoof
    print("False Positive-->The classifier model predicted " + str(
        FP) + " spoof(Negative) samples as real(Positive)")  # Spoof predicted as real
    print("True Negative-->The classifier model predicted " + str(
        TN) + " spoof(Negative) samples as spoof(Negative)")  # Spoofs correctly predicted

    print("Precision of the Linear SVM:"
          , (TP / (TP + FP)))
    print("Recall of the Linear SVM:", (TP / (TP + FN)))
    print("Accuracy of the Linear SVM:", ((TP + TN) / (TP + TN + FP + FN)))

    far = FP / (FP + TN)  # apcer
    frr = FN / (TP + FN)  # bpcer
    hter = (far + frr) / 2  # hter

    fpr, tpr, threshold = metrics.roc_curve(labels2, predict_proba[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    dist = abs((1 - fpr) - tpr)
    eer = fpr[np.argmin(dist)]
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("/Users/josephmarcus/Desktop/Thesis/performance/roc.png")
    return far, frr, hter, eer
