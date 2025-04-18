#读取csv文件,获取target，predict,并绘制混淆矩阵，计算准确率，召回率，F1值，auc值，精确率
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
#读取csv文件
df = pd.read_csv('job_name_predicted_outputs.csv')
#获取target和predict
target = df['target']
predict = df['prediction']

#将target和predict转换为numpy数组
target = np.array(target)
predict = np.array(predict)

#计算准确率，召回率，F1值，auc值，精确率
auc = roc_auc_score(target, predict)
predict = predict > 0.5
#predict = predict.astype(int)
accuracy = accuracy_score(target, predict)
recall = recall_score(target, predict)
f1 = f1_score(target, predict)
precision = precision_score(target, predict)
#输出准确率，召回率，F1值，auc值，精确率
print('准确率：', accuracy)
print('精确率：', precision)
print('召回率：', recall)
print('F1值：', f1)
print('auc值：', auc)

#绘制混淆矩阵
cm = confusion_matrix(target, predict)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')

