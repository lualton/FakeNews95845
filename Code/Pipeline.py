import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def multi_fun_roc_cf(x, y, clf, Method_Name, class_name, data_source='unknown', cross_val=10):
    tprs = []
    aucs = []
    mean_accuracy = 0
    mean_confusion_matrix = np.zeros((len(class_name), len(class_name)))
    mean_fpr = np.linspace(0, 1, 100)
    clf_rf = clf
    cv = StratifiedKFold(10)
    for train_index, test_index in cv.split(x, y):
        prediction_model = clf_rf.fit(x[train_index], y[train_index])
        prediction_probs = prediction_model.predict_proba(x[test_index])
        prediction = prediction_model.predict(x[test_index])
        accuracy = accuracy_score(y[test_index], prediction)
        mean_accuracy += accuracy
        mean_confusion_matrix += confusion_matrix(y[test_index], prediction)
        fpr, tpr, thresholds = roc_curve(y[test_index], prediction_probs[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_accuracy /= cross_val
    print(Method_Name + 'mean accuracy', mean_accuracy)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_confusion_matrix = mean_confusion_matrix.astype(np.int16)

    plt.figure(figsize=(10, 5))
    plt.suptitle(Method_Name + r" accuracy : %.4f" % mean_accuracy, fontsize=16)
    plt.subplot(121)
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=1)
    plt.legend([r'average roc plot with mean auc %0.2f' % (mean_auc)])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve')
    plt.subplot(122)
    plot_confusion_matrix(mean_confusion_matrix, classes=['True', 'Fake'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(r"%s_%s.png" % (data_source, Method_Name))
    plt.clf()
    plt.close()
    # plt.show()


def simple_prediction(x, y, clf, Method_Name, class_name, data_source='unknown', cross_val=10):
    mean_accuracy = 0
    mean_confusion_matrix = np.zeros((len(class_name), len(class_name)))
    clf_rf = clf
    cv = StratifiedKFold(10)
    for train_index, test_index in cv.split(x, y):
        prediction_model = clf_rf.fit(x[train_index], y[train_index])
        prediction = prediction_model.predict(x[test_index])
        accuracy = accuracy_score(y[test_index], prediction)
        mean_accuracy += accuracy
        mean_confusion_matrix += confusion_matrix(y[test_index], prediction)

    mean_accuracy /= 10
    print(Method_Name + r' mean accuracy %.4f' % mean_accuracy)
    mean_confusion_matrix = mean_confusion_matrix.astype(np.int16)
    plt.suptitle(Method_Name + ' Accuracy:' + str(mean_accuracy))
    plot_confusion_matrix(mean_confusion_matrix, classes=['True', 'Fake'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(r"%s_%s_simple.png" % (data_source, Method_Name))
    plt.clf()
    plt.close()
    # plt.show()


def first_pipe(csv_files):
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        try:
            data = data.drop(['dataset'], axis=1)
        except:
            pass

        # prepare x and y
        x = data.loc[:, data.columns != 'class']
        x = np.array(x)
        y = data.loc[:, data.columns == 'class']
        y = np.array(y).ravel()
        y_encode = y == 'fake'
        class_name = ['fake', 'true']

        clf = RandomForestClassifier(n_estimators=10, n_jobs=4, criterion='entropy')
        Method_Name = 'Random Forest'
        multi_fun_roc_cf(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name, data_source=csv_file)

        clf = GaussianNB()
        Method_Name = 'Naive Bayes'
        multi_fun_roc_cf(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name, data_source=csv_file)

        clf = KNeighborsClassifier(n_neighbors=3)
        Method_Name = '3-Nearest'
        simple_prediction(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name,
                          data_source=csv_file)
        multi_fun_roc_cf(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name, data_source=csv_file)

        clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=10, n_jobs=4, criterion='entropy'),
                                 n_estimators=10)
        Method_Name = 'Adaboost_RandomForest'
        multi_fun_roc_cf(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name, data_source=csv_file)

        clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=5), n_estimators=10)
        Method_Name = 'Adaboost_DecisionTree'
        multi_fun_roc_cf(x=x, y=y_encode, clf=clf, Method_Name=Method_Name, class_name=class_name, data_source=csv_file)


# read data and drop unnecessary
# csv_file = './data_expandedfeatures_balanced_20compress.csv'
# csv_file = './new_combined_data_balanced_20compress.csv'
# csv_file = './data_Wang_expandedfeatures_balanced_20compress.csv'

csv_files = ['./data_expandedfeatures_balanced_20compress.csv', './new_combined_data_balanced_20compress.csv',
             './data_Wang_expandedfeatures_balanced_20compress.csv']
csv_files = ['new_combined_data_balanced_50compress.csv']
csv_files = ['new_combined_data_balanced_20compress.csv']
csv_files = ['./data_expandedfeatures_balanced_20compress.csv']
csv_files = ['./data_Wang_expandedfeatures_balanced_20compress.csv']

first_pipe(csv_files)
