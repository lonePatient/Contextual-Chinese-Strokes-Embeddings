#encoding:utf-8
import torch
from ..model.nn.language_model import LanguageModel

class LMEmbedding(object):
    '''
    输入一句话，返回每个词的词向量
    '''
    def __init__(self,
                 model_path,
                 data_transform,
                 device):

        self.device = device
        self.model_path = model_path
        self.data_transform = data_transform
        self.load_language_model()

    def load_language_model(self):
        '''
        加载语言模型
        :param model_path:
        :return:
        '''
        state = torch.load(str(self.model_path))
        model_configs = state['model_configs']
        self.is_forward_lm = model_configs['is_forward_lm']
        self.model = LanguageModel(model_configs= model_configs)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)

    def to_tokenized_string(self,sentence):
        '''
        以空格方式分割字符
        :param sentence:
        :return:
        '''
        sentence = " ".join(list("".join(sentence.strip('\n').strip().split())))
        return sentence

    def get_tokens(self,sentence):
        '''
        获取所有tokens
        :param sentence:
        :return:
        '''
        tokens = sentence.split(" ")
        return tokens

    def get_representation(self,text_sentences,chars_per_chunk = 512):
        strokes_sentences = []
        for sentence in text_sentences:
            stroke_list = [stroke for word in list(sentence) for stroke in
                           self.data_transform.get_strokes_for_word(word)]
            strokes_sentences.append(stroke_list)
        # 列表中，一句话的最大长度（按照字符计算的）
        longest_character_sequence_in_batch: int = len(max(strokes_sentences, key=len))
        # pad strings with whitespaces to longest sentence
        # 空白嵌入
        sentences_padded = []
        marker = ' '
        # 遍历句子
        for sentence_text in strokes_sentences:
            pad_by = longest_character_sequence_in_batch - len(sentence_text)
            if self.is_forward_lm:
                # 前向
                padded = [marker]+sentence_text + [marker] + [' '] * pad_by
                sentences_padded.append(padded)
            else:
                # 反向
                padded =[marker]+sentence_text[::-1] + [marker] + [' '] * pad_by
                sentences_padded.append(padded)

        # cut up the input into chunks of max charlength = chunk_size
        longest = len(sentences_padded[0])  # 句子的最大长度
        chunks = []
        splice_begin = 0
        # 如果超过chars_per_chunk，则按照chars_per_chunk长度进行分割句子
        for splice_end in range(chars_per_chunk, longest, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in sentences_padded])
            splice_begin = splice_end
        # 剩余部分
        chunks.append([text[splice_begin:longest] for text in sentences_padded])
        output_parts = []
        # 遍历每个chunks
        # push each chunk through the RNN language model
        for chunk in chunks:
            sequences_as_char_indices = []
            # 遍历字符
            for string in chunk:
                # 获取该句子的字符串表示
                char_indices = [self.data_transform.get_idx_for_item(char) for char in string]
                sequences_as_char_indices.append(char_indices)
            # transpose: 变换纬度，该例子中，类似于:2x3 -> 3x2
            # 转化为列模式
            batch = torch.LongTensor(sequences_as_char_indices).transpose(0, 1)
            batch = batch.to(self.device)
            # 模型的输出
            prediction, rnn_output, hidden = self.model(batch)
            # 分离出：rnn_output表示序列状态的输出
            rnn_output = rnn_output.detach()
            output_parts.append(rnn_output)
        # concatenate all chunks to make final output
        output = torch.cat(output_parts)
        return output

    def computer_embedding(self, sentences):
        '''
        计算词向量
        :param sentences:
        :return:
        '''
        # make compatible with serialized models
        if 'chars_per_chunk' not in self.__dict__:
            self.chars_per_chunk = 512
        embedding_dict = {}
        with torch.no_grad():
            # if this is not possible, use LM to generate embedding. First, get text sentences
            # 转化为文本形式
            text_sentences = [self.to_tokenized_string(sentence) for sentence in sentences]
            marker = ' '
            extra_offset = len(marker)
            # 模型的隐藏状态输出，输入的是一个句子
            # get hidden states from language model
            all_hidden_states_in_lm = self.get_representation(text_sentences, self.chars_per_chunk)

            # take first or last hidden states from language model as word representation
            for i, sentence_text in enumerate(text_sentences):
                offset_forward = extra_offset
                offset_backward = len(sentence_text) + extra_offset
                words = self.get_tokens(sentence_text)
                embedding_dict[i] = {}
                for word in words:
                    # 前向，前几个词表示当前词
                    offset_forward += len(self.data_transform.get_strokes_for_word(word))
                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward
                    embedding = all_hidden_states_in_lm[offset, i, :]
                    # if self.tokenized_lm or token.whitespace_after:
                    # 词与词之间是按照空格进行分割的，因此，需要减去1或者加1
                    offset_forward += 1
                    offset_backward -= 1
                    offset_backward -= len(word)
                    # 返回一个词的词向量
                    embedding_dict[i][word] = embedding.clone().detach().cpu()
        return embedding_dict