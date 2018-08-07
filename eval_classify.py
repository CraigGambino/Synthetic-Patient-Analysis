import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
np.set_printoptions(suppress=True, precision=4)

def binary_auc_gen(y_test, y_score, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=0)
    area = auc(tpr, fpr)

    plt.figure(figsize=(12,8))
    lw=2
    plt.plot(tpr, fpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % area)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def confusion_matrix_plot(cm, title='Confusion Matrix', context='notebook'):
    sns.set_context(context=context, font_scale=1.5)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='RdBu', cbar=False)
    np.set_printoptions(suppress=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['Positive', 'Negative'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    plt.show()

def binary_rundown(cm):
    true_pos = cm[0][0]
    true_neg = cm[1][1]
    false_pos = cm[1][0]
    false_neg = cm[0][1]
    specificity = true_neg / (true_neg + false_pos)
    sensitivity = true_pos / (true_pos + false_neg)
    ppv = true_pos / (true_pos + false_pos)
    npv = true_neg / (true_neg + false_neg)
    print(
    ' Specificity: {}'.format(specificity),
    '\n Sensitivity: {}'.format(sensitivity),
    '\n Positive Predicitve Value: {}'.format(ppv),
    '\n Negative Predicitve Value: {}'.format(npv)
    )
