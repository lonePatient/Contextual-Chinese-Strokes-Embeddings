#encoding:utf-8
from pathlib import Path
from pyLM.utils.logginger import init_logger
from pyLM.test.embedding import LMEmbedding
from pyLM.io.data_transformer import DataTransformer
from pyLM.config.basic_config import configs as config

# **************************** 基础信息 ***********************
logger = init_logger(log_name=config['arch'], log_dir=config['output']['log_dir'])
device = 'cuda:%d' % config['train']['n_gpu'][0] if len(config['train']['n_gpu']) else 'cpu'


model_path = Path(config['output']['checkpoint_dir']) / config['callback']['best_model_name'].format(arch = config['arch'])
data_transformer = DataTransformer(logger=logger,
                                   chars_path=config['data']['chars_path'],
                                   strokes_path=config['data']['strokes_path'],
                                   mapping_path=config['data']['mapping_path'],
                                   word_stroke_path=config['data']['word_stroke_path'],
                                   add_unk=True)

sentence = '学莉娜搭配做可爱日系美人第九天机车皮衣搭配豹纹抹胸裙'
lm_embeddings = LMEmbedding(device = device,
                                model_path = model_path,
                                data_transform = data_transformer)
stroke_embeddings = lm_embeddings.computer_embedding(sentences=[sentence])
for i,sent in enumerate([sentence]):
    for word in list(sent):
        print(word)
        print(stroke_embeddings[i][word])