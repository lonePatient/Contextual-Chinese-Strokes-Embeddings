#encoding:utf-8
import os
import math
import time
import torch
import numpy as np
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint,model_device


class Trainer(object):
    '''
    语言模型训练器
    '''
    def __init__(self,model,
                 train_data,
                 valid_data,
                 optimizer,
                 epochs,
                 logger,
                 clip_grad,
                 ntokens,
                 batch_size,
                 sequence_length,
                 criterion,
                 evaluate,
                 lr_scheduler,
                 n_gpu            = None,
                 resume           = None,
                 model_checkpoint = None,
                 training_monitor = None,
                 early_stopping   = None,
                 verbose = 1):

        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.resume = resume
        self.ntokens = ntokens
        self.n_gpu = n_gpu
        self.clip_grad = clip_grad
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.verbose = verbose
        self.evaluate = evaluate
        self.criterion = criterion
        self.batch_size = batch_size
        self.training_monitor = training_monitor
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.lr_scheduler = lr_scheduler
        self.sequence_length = sequence_length
        self._reset()

    def _reset(self):
        '''
        重置信息
        :return:
        '''
        self.start_epoch = 1
        self.global_step = 0
        self.batch_num = len(self.train_data)
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)

        # 重载模型，进行训练
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch = arch),
                                       self.model_checkpoint.best_model_name.format(arch = arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path = resume_path,model = self.model,optimizer = self.optimizer)
            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            best = resume_list[2]
            self.start_epoch = resume_list[3]

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        '''
        模型的整体结构信息
        :return:
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('trainable parameters num: {}'.format(params))
        # 模型结构
        self.logger.info(self.model)

    def save_info(self,epoch,valid_loss):
        '''
        模型保存信息
        :param epoch:
        :param valid_loss:
        :return:
        '''
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'valid_loss': round(valid_loss,4),
            'model_configs': self.model.model_configs
        }
        return state

    def get_batch(self,source, i, sequence_length):
        '''
        对每个数据文件，我们获取batch大小的数据
        :param source:
        :param i:
        :param sequence_length:
        :return:
        '''
        seq_len = min(sequence_length, len(source) - 1 - i)
        # 取前sequence_length个词预测下一个词
        data = source[i:i + seq_len].clone().detach()
        # 下一个词
        target = source[i + 1:i + 1 + seq_len].view(-1).clone().detach()
        data = data.to(self.device)
        target = target.to(self.device)
        return data, target

    def batchify(self,data):
        '''
        构建batch数据集，这里我们对每个文件构建batch数据集
        注意：当数据量不足batch大小时，这里我们选择丢弃。
        :param data:
        :return:
        '''
        # 计算batch的个数
        nbatch = data.size(0) // self.batch_size
        # 保留nbatch * batch_size数据，不足batch size数据丢弃
        data = data.narrow(0, 0, nbatch * self.batch_size)
        # view：把data转化为(batch_size,-1)的张量，-1表示自动计算纬度
        data = data.view(self.batch_size, -1).t().contiguous()
        return data

    def valid_epoch(self,valid_slice):
        '''
        valid数据集评估
        :return:
        '''
        valid_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i in range(0, valid_slice.size(0) - 1, self.sequence_length):
                data, targets = self.get_batch(valid_slice, i, self.sequence_length)
                prediction, rnn_output, hidden = self.model.forward(data)
                output_flat = prediction.view(-1, self.ntokens)
                loss = self.criterion(output_flat,targets)
                valid_loss.update(loss.item(),data.size(0))
        return {
            'valid_loss': valid_loss.avg,
            'valid_ppl': math.exp(valid_loss.avg)
        }

    def train_epoch(self,train_slice):
        '''
        train数据集训练过程
        :param train_slice:
        :return:
        '''
        train_loss = AverageMeter()
        self.model.train()
        for step, i in enumerate(range(0,train_slice.size(0)-1,self.sequence_length)):
            start = time.time()
            data,target = self.get_batch(train_slice,i,self.sequence_length)
            # self.model.zero_grad()
            output,rnn_output,hidden = self.model(data)
            self.optimizer.zero_grad()
            loss = self.criterion(output.view(-1,self.ntokens),target)
            loss.backward()
            # 防止RNN/LSTM中出现梯度爆炸问题
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_grad)
            self.optimizer.step()
            train_loss.update(loss.item(), n=1)
            # explicitly remove loss to clear up memory
            del loss, output, rnn_output
            # 实时打印训练过程
            if self.verbose >= 1:
                self.progressbar.batch_step(batch_idx= step,
                                            info = {'loss':train_loss.avg,
                                                    'ppl':math.exp(train_loss.avg)},
                                            use_time = time.time() - start)
        print(" ")
        train_log = {
            'loss': train_loss.avg,
            'ppl': math.exp(train_loss.avg),
        }
        return train_log

    def train(self):
        '''
        整个模型的训练过程
        :return:
        '''
        # 这里稍微注意下，valid_data只有一个数据文件
        for step, valid_slice in enumerate(self.valid_data):
            valid_slice = self.batchify(data=valid_slice.flatten())

        print("----------------- training start -----------------------")
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch + self.epochs - 1))
            for curr_split, train_slice in enumerate(self.train_data):
                train_slice = self.batchify(data = train_slice.flatten())
                self.global_step += 1
                self.progressbar = ProgressBar(n_batch=len(list(range(0,train_slice.size(0)-1,self.sequence_length))))

                train_log = self.train_epoch(train_slice)
                valid_log = self.valid_epoch(valid_slice)

                logs = dict(train_log,**valid_log)
                self.lr_scheduler.epoch_step(logs['valid_loss'],epoch = epoch)
                show_info = '\nStep: %d - '%self.global_step + "-".join([' %s: %.4f '%(key,value) for key,value in  logs.items()])
                self.logger.info(show_info)

                if self.training_monitor:
                    self.training_monitor.step(logs)

                if self.model_checkpoint:
                    state = self.save_info(epoch,logs['valid_loss'])
                    self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)

                if self.early_stopping:
                    self.early_stopping.step(epoch=self.global_step, current=logs[self.early_stopping.monitor])
                    if self.early_stopping.stop_training:
                        break
