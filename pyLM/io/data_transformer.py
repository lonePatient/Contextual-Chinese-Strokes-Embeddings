#encoding:utf-8
from ..utils.utils import pkl_write,pkl_read

class DataTransformer(object):
    '''
    构建stroke映射字典，以及词与stroke的映射字典
    '''
    def __init__(self,
                 logger,
                 chars_path,
                 strokes_path,
                 mapping_path,
                 word_stroke_path,
                 add_unk = True
                 ):
        # 初始化字典
        self.logger = logger
        self.strokes_path = str(strokes_path)
        self.item2idx = {} # stroke对应的id
        self.idx2item = [] # item列表
        self.word2strokes = {} # 每个词对应的strokes
        # 未知的tokens
        if add_unk:
            self.add_item('<unk>')
        self.read_chars(chars_path=chars_path)
        self.read_word_strokes(mapping_path,word_stroke_path)

    def add_item(self,item):
        '''
        对映射字典中新增item
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1

    def get_idx_for_item(self,item):
        '''
        获取指定item的id，如果不存在，则返回0，即unk
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return 0

    def get_item_for_index(self, idx):
        '''
        给定id，返回对应的tokens
        :param idx:
        :return:
        '''
        return self.idx2item[idx].decode('UTF-8')

    def get_items(self):
        '''
        获取所有的items
        :return:
        '''
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))

    def get_strokes_for_word(self,word):
        '''
        获取给订word的所有的strokes信息，如果未在映射表中，则
        返回词本身，正常情况一般为各种符号，中英文和数字等
        :param word:
        :return: typre list
        '''
        if word in self.word2strokes:
            return self.word2strokes[word]
        else:
            return [word]

    def read_chars(self,chars_path):
        '''
        读取字符串列表，包括笔画、符号以及数字、英文等
        :param chars_path:
        :return:
        '''
        chars = pkl_read(chars_path)
        for char in chars:
            self.add_item(char)
        self.logger.info("all {} tokens".format(len(self.idx2item)))

    def read_word_strokes(self,mapping_path,word_stroke_path):
        '''
        读取strokes原始数据信息,原文件数据格式:
        Examples:
            矩:撇,横,横,撇,点,横,横折,横,竖折
            顼:横,横,竖,提,横,撇,竖,横折,撇,点
            ...
        '''
        with open(self.strokes_path,'r') as fr:
            for i,line in enumerate(fr):
                line = line.strip('\n') # 去除换行符了
                lines = line.split(":") # 文件默认使用:进行分割
                word = lines[0]   # word
                info = lines[1].split(",") # strokes中使用","进行分割
                # 每一个中文对应的strokes信息
                self.word2strokes[word] = info
        mappings = {
            'idx2item': self.idx2item,
            'item2idx': self.item2idx
        }
        # 将数据写入文件中
        pkl_write(filename = mapping_path,data = mappings)
        pkl_write(filename = word_stroke_path,data = self.word2strokes)



