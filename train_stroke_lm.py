#encoding:utf-8
import torch
import warnings
from torch.utils.data import DataLoader
from pyLM.train.losses import CrossEntropy
from pyLM.train.trainer import Trainer
from pyLM.io.dataset import CreateDataset
from pyLM.io.data_transformer import DataTransformer
from pyLM.utils.logginger import init_logger
from pyLM.utils.utils import seed_everything
from pyLM.config.basic_config import configs as config
from pyLM.model.nn.language_model import LanguageModel
from pyLM.callback.optimizater import AdamW
from pyLM.callback.lrscheduler import ReduceLRWDOnPlateau
from pyLM.callback.modelcheckpoint import ModelCheckpoint
from pyLM.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

# 主函数
def main():
    # **************************** 基础信息 ***********************
    logger = init_logger(log_name = config['arch'], log_dir = config['output']['log_dir'])
    logger.info("seed is %d"%config['train']['seed'])
    device = 'cuda:%d' % config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'
    seed_everything(seed = config['train']['seed'],device = device)
    logger.info('starting load data from disk')
    # **************************** 数据生成 ***********************

    data_transformer = DataTransformer(logger = logger,
                                       chars_path = config['data']['chars_path'],
                                       strokes_path = config['data']['strokes_path'],
                                       mapping_path = config['data']['mapping_path'],
                                       word_stroke_path = config['data']['word_stroke_path'],
                                       add_unk = True)
    # train
    train_dataset   = CreateDataset(logger = logger,
                                    data_path = config['data']['train_file_dir'],
                                    data_transform = data_transformer,
                                    is_forward_lm = config['model']['is_forward_lm'],
                                    shuffle_lines = True)

    valid_dataset   = CreateDataset(logger = logger,
                                    data_path = config['data']['valid_file_path'],
                                    data_transform = data_transformer,
                                    is_forward_lm = config['model']['is_forward_lm'],
                                    shuffle_lines = False)

    train_generator = DataLoader(dataset = train_dataset,
                                 num_workers = config['train']['num_workers'],
                                 shuffle = True,
                                 drop_last = False,
                                 pin_memory = False)

    valid_generator = DataLoader(dataset = valid_dataset,
                                num_workers = config['train']['num_workers'],
                                shuffle = False,
                                drop_last = False,
                                pin_memory = True)

    ntokens = len(data_transformer.item2idx)
    # **************************** 模型 ***********************
    logger.info("initializing model")
    model_configs = config['model']
    model_configs['input_size'] = ntokens
    model_configs['embedding_size'] = config['train']['embedding_size']
    model = LanguageModel(model_configs = config['model'])

    # ************************** 优化器 *************************
    optimizer = AdamW(model.parameters(),
                      lr = config['train']['learning_rate'],
                      weight_decay = config['train']['weight_decay'])

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir = config['output']['checkpoint_dir'],
                                       mode = config['callback']['mode'],
                                       monitor = config['callback']['monitor'],
                                       save_best_only = config['callback']['save_best_only'],
                                       best_model_name = config['callback']['best_model_name'],
                                       epoch_model_name = config['callback']['epoch_model_name'],
                                       arch = config['arch'],
                                       logger = logger)
    # 监控训练过程
    train_monitor = TrainingMonitor(file_dir = config['output']['figure_dir'],
                                    arch = config['arch'],
                                    start_at= 0 )
    # 学习率机制
    lr_scheduler = ReduceLRWDOnPlateau(optimizer = optimizer,
                                       factor = 0.5,
                                       patience = config['callback']['lr_patience'],
                                       min_lr = 1e-8,
                                       verbose = 1,
                                       mode = config['callback']['mode'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model = model,
                      logger=logger,
                      ntokens = ntokens,
                      evaluate=None,
                      early_stopping=None,
                      train_data = train_generator,
                      valid_data = valid_generator,
                      optimizer = optimizer,
                      epochs = config['train']['epochs'],
                      criterion = CrossEntropy(),
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      resume = config['callback']['resume'],
                      lr_scheduler = lr_scheduler,
                      n_gpu = config['train']['n_gpu'],
                      batch_size = config['train']['batch_size'],
                      sequence_length=config['train']['sequence_length'],
                      clip_grad = config['train']['clip_grad'])
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
    # 释放显存
    if len(config['train']['n_gpu']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
