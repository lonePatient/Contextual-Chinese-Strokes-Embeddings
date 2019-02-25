#encodig:utf-8
import torch
import math
import torch.nn as nn

class LanguageModel(nn.Module):
    '''
    基于LSTM结构构建语言模型，
    1、有论文提出，通过共享参数可以减少参数数量，即词向量层和softmax层的大小设置一样，
    从而使用同一个初始化，

    2、如果共享词向量层和softmax层的参数，不仅能大幅减少参数数量，还能提高最终模型的效果。
    （需要做实验进行验证）
    '''
    def __init__(self,
                 model_configs):
        super(LanguageModel,self).__init__()
        self.model_configs = model_configs

        self.input_size = model_configs['input_size']
        self.dropout = model_configs['dropout']
        self.num_layers = model_configs['num_layers']
        self.hidden_size = model_configs['hidden_size']
        self.embedding_size = model_configs['embedding_size']
        self.out_num = model_configs['out_num'] # 是否指定全连接层的大小

        self.drop = nn.Dropout(model_configs['dropout'])
        self.encoder = nn.Embedding(num_embeddings = model_configs['input_size'],
                                    embedding_dim = model_configs['embedding_size'])

        # LSTM 参数说明:
        #   num_layers: 表示几层的循环网络，默认为1
        #   dropout：除了最后一层之外都引入一个dropout
        #   batch_first：默认为False，因为nn.lstm接受的数据输入时(seq_len,batch_size,embed_dize),
        #   如果为True，则转化为(batch_szie,seq_len.embed_size)
        #   input_size: 输入数据的特征数
        #   hidden_size：输出的特征数(hidden_size)

        # 使用LSTM网络的使用方式，每一层LSTM都有三个外界的输入数据，即:
        #   X；LSTM网络输入的数据
        #   h_0: 上一层LSTM输出的结果
        #   c_0: 上一层LSTM调整后的记忆
        #   一般而言h_0和c_0初始化为0
        if self.num_layers == 1:
            self.rnn = nn.LSTM(input_size = self.embedding_size,
                               hidden_size = self.hidden_size,
                               num_layers = self.num_layers)
        else:
            self.rnn = nn.LSTM(input_size = self.embedding_size,
                               hidden_size = self.hidden_size,
                               num_layers = self.num_layers,
                               dropout = self.dropout)

        # LSTM的输出为（seq_len,batch,hidden_size * num_directions),
        # h_n: (num_layers * num_directions , batch_size, hidden_size)
        # c_n: (num_layers * num_directions , batch_size, hidden_size)

        if self.out_num is not None:
            self.proj = nn.Linear(in_features = self.hidden_size,out_features = self.out_num)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(in_features = self.out_num,
                                     out_features = self.input_size)
        else:
            self.decoder = nn.Linear(in_features = self.hidden_size,
                                     out_features = self.input_size)
        # 初始化参数
        self.init_weights()

    def initialize(self,matrix):
        '''
        初始化全连接层权重
        :param matrix:
        :return:
        '''
        in_,out_ = matrix.size()
        stdv = math.sqrt(3. / (in_ + out_))
        matrix.detach().uniform_(-stdv,stdv)

    def init_weights(self):
        '''
        初始化编码器和解码器权重
        :return:
        '''
        initrange = 0.1
        self.encoder.weight.detach().uniform_(-initrange,initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_(-initrange,initrange)

    def forward(self,input):
        '''
        前向计算
        :param input:
        :return:
        '''
        encoder = self.encoder(input)
        emb = self.drop(encoder)
        self.rnn.flatten_parameters()
        # lstm输出的shape为:
        # output: (seq_len, batch_size, hidden_size * num_directions
        # hidden: (h_n,c_n)
        # h_n: (num_layers * num_directions , batch_size, hidden_size)
        # c_n: (num_layers * num_directions , batch_size, hidden_size)
        output,hidden = self.rnn(emb)
        output = self.drop(output)
        # 对于Linear层，我们需要将数据shape转化为(seq_len * batch_size,hidden_size)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),output.size(2)))
        # 重新将output的shape转换为(seq_len,batch_size,hidden_size)
        return decoded.view(output.size(0),output.size(1),decoded.size(1)),output,hidden
