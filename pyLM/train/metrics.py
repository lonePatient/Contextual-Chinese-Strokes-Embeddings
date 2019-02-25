#encoding:utf-8
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report

class Accuracy(object):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

class ClassReport(object):
    '''
    类别分类报告
    '''
    def __init__(self,target_names = None):
        self.target_names = target_names

    def __call__(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        classify_report = classification_report(y_true, y_pred,target_names=self.target_names)
        print('\n\nclassify_report:\n', classify_report)


class F1Score(object):
    '''
    由于使用的是F1得分，因此我们需要找到一个最佳的thresh值来确定class类别，因此一般而言需要对thresh值进行优化。
    '''
    def __init__(self,thresh = 0.5,normalizate = True,task_type = 'binary'):
        assert task_type in ['binary','multiclass']
        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.reset()

    def reset(self):
        self.outputs = None
        self.targets = None

    # 对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
    # 这里我们队Thresh进行优化
    def thresh_search(self,y_true,y_proba):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            y_pred = y_proba > threshold
            score = self._compute(output=y_pred,target=y_true)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    # 计算指标得分
    def _compute(self,output,target):
        if self.task_type == 'binary':
            f1 =f1_score(y_true = target, y_pred=output,average='binary')
            return f1
        if self.task_type == 'multiclass':
            f1 = f1_score(y_true=target, y_pred=output, average='macro')
            return f1

    # 计算整个结果
    def result(self):
        if self.task_type == 'binary':
            if self.thresh:
                outputs = (self.outputs[:,1] > self.thresh ).astype(int)
                f1 = self._compute(output= outputs, target=self.targets)
                print("F1 thresh: %.4f - Score: %.4f"%(self.thresh,f1))
            else:
                threshold,f1 = self.thresh_search(y_true = self.targets,y_proba=self.outputs[:,1])
                print("best thresh: %.4f - F1 Score: %.4f" % (threshold, f1))

        if self.task_type == 'multiclass':
            outputs = np.argmax(self.outputs, 1)
            f1 = self._compute(output=outputs, target=self.targets)
            print("F1 Score: %.4f" % (f1))

    def update(self,output,target):
        if self.normalizate and self.task_type == 'binary':
            y_pred = output.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_pred = output.softmax().data.cpu().detach().numpy()
        else:
            y_pred = output.cpu().detach().numpy()
        y_true = target.cpu().numpy()

        if self.outputs is None:
            self.outputs = y_pred
            self.targets = y_true
        else:
            self.outputs = np.concatenate((self.outputs,y_pred),axis =0)
            self.targets = np.concatenate((self.targets, y_true), axis=0)