#encoding:utf-8
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    '''
    一般而言，我们使用大的语料库进行语言模型的训练，
    通常会把训练集划分成好几份(太大无法加载到内存中)。因此
    文件结构类似以下格式。
    Examples:

    pyLM/dataset/raw/train/train_01.txt
    pyLM/dataset/raw/train/train_02.txt
    ...
    pyLM/dataset/raw/valid.txt
    pyLM/dataset/raw/test.txt

    '''
    def __init__(self,
                 logger,
                 data_path,
                 data_transform,
                 is_forward_lm = True,
                 random_case_flip = False, #中文不起作用，因此默认为False
                 shuffle_lines = True):

        if isinstance(data_path,Path):
            pass
        else:
            data_path = Path(data_path)
        assert data_path.exists()

        self.logger = logger
        self.data_path = data_path
        self.data_transform = data_transform # 数据映射字典
        self.random_case_flip = random_case_flip
        self.shuffle_lines = shuffle_lines # 行之间打乱数据
        self.is_forward_lm = is_forward_lm  # 是否前向语言模型

        if data_path.is_dir():
            # 自动获取训练数据集分片文件列表
            self.files = sorted([f for f in data_path.iterdir() if f.exists()])
        else:
            # valid和test数据集
            self.files = [data_path]

    def random_casechange(self,line):
        '''
        大小写变换，只有在英文语料下才起作用
        :param line:
        :return:
        '''
        no = random.randint(0, 99)
        if no is 0:
            line = line.lower()
        if no is 1:
            line = line.upper()
        return line

    def build_features(self,lines):
        '''
        将数据转化为特征，id化以及构造输入data、target数据
        :param lines:
        :return:
        '''
        if self.shuffle_lines:
            random.shuffle(lines)
            # self.logger.info('shuffled')
        tokens = 0
        # 这里我们将中文句子转化为以空格为分割的字列表，(按照英文格式）即
        # Example:
        # raw: 而且资讯都是公开的，可以在游戏版上确认战斗能否胜利
        # new: 而 且 资 讯 都 是 公 开 的 ， 可 以 在 游 戏 版 上 确 认 战 斗 能 否 胜 利
        # 这里稍微注意下："".join(line.split())主要是想去除\xa0符号
        lines = [" ".join(list("".join(line.strip().split()))) for line in lines]
        # 计算数据大小
        # for line in lines:
        #     for word in list(line):
        #         strokes = self.data_transform.get_strokes_for_word(word)
        #         tokens += len(strokes)
        tokens = sum([len(self.data_transform.get_strokes_for_word(word)) for line in lines for word in list(line)])
        self.logger.info("compute text file with {} tokens".format(tokens))
        # 初始化一个ids矩阵，存放数据id
        ids = torch.zeros(tokens,dtype = torch.long)

        # 前向语言模型
        if self.is_forward_lm:
            token = 0
            for line in lines:
                # 对中文不起任何作用,所以这里为False
                if self.random_case_flip:
                    line = self.random_casechange(line)
                stroke_list = [stroke for word in list(line) for stroke in
                               self.data_transform.get_strokes_for_word(word)]
                for stroke in stroke_list:
                    if token >= tokens:break
                    ids[token] = self.data_transform.get_idx_for_item(stroke)
                    token += 1

        # 后向语言模型
        else:
            token = tokens - 1
            for line in lines:
                # 对中文不起作用
                if self.random_case_flip:
                    line = self.random_casechange(line)
                stroke_list = [stroke for word in list(line) for stroke in
                                self.data_transform.get_strokes_for_word(word)]
                for stroke in stroke_list:
                    if token >= tokens:break
                    ids[token] = self.data_transform.get_idx_for_item(stroke)
                    token -= 1
        return ids

    def preprocess(self,index):
        '''
        主函数
        :param index:
        :return:
        '''
        # 根据index返回数据file_path
        path = self.files[index]
        assert path.exists()
        # 读取数据
        lines = open(path,'r',encoding = 'UTF-8').readlines()
        self.logger.info('read text file with {} lines'.format(len(lines)))
        # 将text转化ids数据
        ids = self.build_features(lines)
        return ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.preprocess(index)


