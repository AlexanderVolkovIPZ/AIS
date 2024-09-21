import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_curve, roc_auc_score
import math
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
df.head()

confusion_matrix(df.actual_label.values, df.predicted_RF.values)

def find_TP(y_true, y_pred):
 # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
 # counts the number of false negatives (y_true = 1, y_pred =0)
 return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
 # counts the number of false positives (y_true = 0, y_pred =1)
 return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
 # counts the number of true negatives (y_true = 0, y_pred =0)
 return sum((y_true == 0) & (y_pred == 0))

print('TP:',find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:',find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:',find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:',find_TN(df.actual_label.values, df.predicted_RF.values))

def find_conf_matrix_values(y_true,y_pred):
 # calculate TP, FN, FP, TN
 TP = find_TP(y_true,y_pred)
 FN = find_FN(y_true,y_pred)
 FP = find_FP(y_true,y_pred)
 TN = find_TN(y_true,y_pred)
 return TP,FN,FP,TN

def volkov_confusion_matrix(y_true, y_pred):
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return np.array([[TN,FP],[FN,TP]])

#Перевірка відповідності результатів зі створеною мною функції volkov_confusion_matrix
print('[TN, FP], \n[FN, TP] \n',volkov_confusion_matrix(df.actual_label.values, df.predicted_RF.values))


assert np.array_equal(volkov_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
confusion_matrix(df.actual_label.values, df.predicted_RF.values)
), 'volkov_confusion_matrix() is not correct for RF'

assert np.array_equal(volkov_confusion_matrix(df.actual_label.values,
df.predicted_LR.values),confusion_matrix(df.actual_label.values,
df.predicted_LR.values) ), 'volkov_confusion_matrix() is not correct for LR'

print("Accurancy_score -> ", accuracy_score(df.actual_label.values, df.predicted_RF.values))

##################################################################
def volkov_accuracy_score(y_true, y_pred):
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return (TP + TN) / (TP + TN + FP + FN)

assert volkov_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values),'volkov_recall_score failed on'
assert volkov_accuracy_score(df.actual_label.values,
df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), 'volkov_recall_score failed on LR'

print('Accuracy RF: %.3f'%(volkov_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f'%(volkov_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

def volkov_recall_score(y_true, y_pred):
 # calculates the fraction of positive samples predicted correctly
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return TP / (TP + FN)

assert volkov_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'volkov_recall_score failed on RF'
assert volkov_recall_score(df.actual_label.values,
df.predicted_LR.values) == recall_score(df.actual_label.values,
df.predicted_LR.values), 'volkov_recall_score failed on LR'
print('Recall RF: %.3f'%(volkov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(volkov_recall_score(df.actual_label.values, df.predicted_LR.values)))

def volkov_precision_score(y_true, y_pred):
 # calculates the fraction of predicted positives samples that are actually positive
 TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
 return TP / (TP + FP)

assert volkov_precision_score(df.actual_label.values,
df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'volkov_accuracy_score failed on RF'
assert volkov_precision_score(df.actual_label.values,
df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'volkov_accuracy_score failed on LR'
print('Precision RF: %.3f'%(volkov_precision_score(df.actual_label.values,  df.predicted_RF.values)))
print('Precision LR:%.3f'%(volkov_precision_score(df.actual_label.values, df.predicted_LR.values)))

def volkov_f1_score(y_true, y_pred):
 # calculates the F1 score
 recall = volkov_recall_score(y_true,y_pred)
 precision = volkov_precision_score(y_true,y_pred)
 return (2.0 * (precision * recall)) / (precision + recall)

# # В мене відбулася похибка на 16 знаку після коми в 1 одиницю тому нижче врахую її
# assert volkov_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values),'volkov_accuracy_score failed on RF'
# assert volkov_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'volkov_accuracy_score failed on LR'

assert math.isclose(
    volkov_f1_score(df.actual_label.values, df.predicted_RF.values),
    f1_score(df.actual_label.values, df.predicted_RF.values),
    rel_tol=1e-15  # допустима відносна похибка
), 'volkov_accuracy_score failed on RF'

assert math.isclose(
    volkov_f1_score(df.actual_label.values, df.predicted_LR.values),
    f1_score(df.actual_label.values, df.predicted_LR.values),
    rel_tol=1e-15  # допустима відносна похибка
), 'volkov_accuracy_score failed on LR'
print('F1 RF: %.3f'%(volkov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f'%(volkov_f1_score(df.actual_label.values, df.predicted_LR.values)))


print('scores with threshold = 0.5')
print('Accuracy RF: %.3f'%(volkov_accuracy_score(df.actual_label.values,df.predicted_RF.values)))
print('Recall RF: %.3f'%(volkov_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(volkov_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(volkov_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(volkov_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(volkov_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(volkov_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(volkov_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))



fpr_RF, tpr_RF,  thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF,'r-',label='RF')
plt.plot(fpr_LR,tpr_LR,'b-', label='LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
