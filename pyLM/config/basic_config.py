#encoding:utf-8
from pathlib import Path
import multiprocessing

BASE_DIR = Path('pyLM')

configs = {
    'arch':'strokes_LM',

    'data':{
        'strokes_path': BASE_DIR / 'dataset/raw/strokes.txt',
        'train_file_dir': BASE_DIR / 'dataset/processed/train', # 一般我们会将一个大数据集分割成好几份，以至于可以加载数据到内存中
        'valid_file_path': BASE_DIR / 'dataset/processed/valid.txt',
        'mapping_path': BASE_DIR / 'dataset/processed/mapping.pkl',
        'word_stroke_path': BASE_DIR / 'dataset/processed/word_stroke.pkl',
        'chars_path': BASE_DIR / 'dataset/raw/chars.pkl'
    },

    'output':{
        'log_dir': BASE_DIR / 'output/log', # 模型运行日志
        'writer_dir': BASE_DIR / "output/TSboard",# TSboard信息保存路径
        'figure_dir': BASE_DIR / "output/figure", # 图形保存路径
        'checkpoint_dir': BASE_DIR / "output/checkpoints",# 模型保存路径
    },

    'pretrained':{

    },

    'train':{
        'sequence_length': 300,
        'num_classes':2,
        'batch_size': 100,  # how many samples to process at once
        'epochs': 100,  # number of epochs to train
        'start_epoch': 1,
        'clip_grad':0.25,
        'embedding_size':300,
        'learning_rate': 0.1,
        'n_gpu': [1], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                       # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller
        'num_workers': multiprocessing.cpu_count(), # 线程个数
        'resume':False,
        'weight_decay': 1e-5,
        'seed':2018,
    },

    'callback':{
        'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',  # 计算指标
        'resume':False,
        'early_patience': 20,   # early_stopping
        'save_best_only': True, # 是否保存最好模型
        'best_model_name': '{arch}-best.pth', # 保存文件
        'epoch_model_name': '{arch}-{epoch}-{info}.pth', # 以epoch频率保存模型
        'save_checkpoint_freq': 10 # 保存模型频率，当save_best_only为False时候，指定才有作用
    },

    'model':{
        "num_layers":1,
        'dropout':0.25,
        'hidden_size':1024,
        'out_num': None,
        'is_forward_lm':True
    }
}
