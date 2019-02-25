#encoding:utf-8
import os
from pathlib import Path
import numpy as np
import torch
from ..utils.utils import ensure_dir

class ModelCheckpoint(object):
    '''
    模型保存，两种模式：
    1. 直接保存最好模型
    2. 按照epoch频率保存模型
    '''
    def __init__(self, checkpoint_dir,
                 monitor,
                 logger,
                 arch,
                 save_best_only = True,
                 best_model_name = None,
                 epoch_model_name = None,
                 mode='min',
                 epoch_freq=1,
                 best = None):

        base_path = Path(str(checkpoint_dir).format(arch=arch))
        base_path.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.logger = logger
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.checkpoint_dir = base_path
        self.save_best_only = save_best_only
        self.best_model_name = best_model_name.format(arch=arch)
        self.epoch_model_name = epoch_model_name

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

    def step(self, state,current):
        # 是否保存最好模型
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.logger.info('\nEpoch %d: %s improved from %0.5f to %0.5f'% (state['epoch'], self.monitor, self.best,current))
                self.best = current
                state['best'] = self.best
                best_path = self.checkpoint_dir / self.best_model_name
                torch.save(state, str(best_path))
        # 每隔几个epoch保存下模型
        else:
            filename = self.checkpoint_dir / self.epoch_model_name.format(arch=self.arch,
                                                                          epoch=state['epoch'],
                                                                          info=state[self.monitor])
            if state['epoch'] % self.epoch_freq == 0:
                self.logger.info("\nEpoch %d: save model to disk."%(state['epoch']))
                torch.save(state, str(filename))
