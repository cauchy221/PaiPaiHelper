import math
import torch
import logging
from torch import nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import MarginRankingLoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from itertools import repeat
# from torch._six import container_abcs
import collections.abc as container_abcs
int_classes = int
string_classes = str

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Modified CNN 
# -------------------------------------------------------------------------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)


class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv1d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def forward(self, input):
        input = input.transpose(1, 2)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)

        output = F.conv1d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        output = output.transpose(1, 2)
        return output


# -------------------------------------------------------------------------------------------
# CnnGram Extractor
# -------------------------------------------------------------------------------------------

class NGramers(nn.Module):
    def __init__(self, input_size, hidden_size, max_gram, dropout_rate):
        super().__init__()

        self.cnn_list = nn.ModuleList([nn.Conv1d(in_channels=input_size,
                                                 out_channels=hidden_size,
                                                 kernel_size=n) for n in range(1, max_gram + 1)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.transpose(1, 2)

        cnn_outpus = []
        for cnn in self.cnn_list:
            y = cnn(x)
            y = self.relu(y)
            y = self.dropout(y)
            cnn_outpus.append(y.transpose(1, 2))
        outputs = torch.cat(cnn_outpus, dim=1)
        return outputs


# -------------------------------------------------------------------------------------------
# Inherit BertPreTrainedModel
# -------------------------------------------------------------------------------------------
class BertForCnnGramKernelRanking(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCnnGramKernelRanking, self).__init__(config)
        max_gram = 5
        cnn_output_size = 512
        cnn_dropout_rate = (config.hidden_dropout_prob / 2)

        self.num_labels = config.num_labels  # ?????????????????????, ??? 5 ???: ????????????, ?????????????????????, ????????????, ?????????, ??????????????????

        self.bert = BertModel(config)
        self.cnn2gram = NGramers(input_size=config.hidden_size,
                                 hidden_size=cnn_output_size,
                                 max_gram=max_gram,
                                 dropout_rate=cnn_dropout_rate)

        self.classifier = nn.Linear(cnn_output_size, 1)  # shape = (512, 1)
        self.chunk_classifier = nn.Linear(cnn_output_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()


# -------------------------------------------------------------------------------------------
# BertForChunkTFRanking
# -------------------------------------------------------------------------------------------
class BertForChunkTFRanking(BertForCnnGramKernelRanking):
    def forward(self, input_ids, attention_mask, valid_ids, active_mask,
                valid_output, labels=None, chunk_labels=None, chunk_mask=None):
        """
        8??????????????? ???????????? batchify_bert2joint_features_for_train ??? ??? 8 ???????????????.
        ??? ?????????, ???????????? 5 ??? ?????? (batchify_bert2joint_features_for_test)
            - input_ids: bert ???????????????????????? ?????? 0-padding ????????????, shape=[batch_size, max_sub_tokens]
            - input_mask: attention_mask ?????? 0-padding ????????????, shape=[batch_size, max_sub_tokens]
            - valid_ids: ??????????????????????????? (valid_mask) ?????? 0-padding ????????????, shape=[batch_size, max_sub_tokens]
            - active_mask: batch???????????????(???index)????????????, ????????????????????????????????????????????????, ????????????????????? 1, ???????????? 0, shape=[batch_size, max_phrases, max_grams]
            - valid_output: ????????????, shape=[len(docs), max_words(??????????????????), bert_output_dim], ?????? ??????/????????? ????????????????????? ?????? ????????????

        ???????????????????????????????????????:
            - label: ??? ngram_label ?????? 0-padding ???????????? ????????????, [ngram_label: phrase_list ??????????????????????????????????????? 1, ???????????? -1]
            - chunk_label: ??? chunk_label ?????? -1-padding ????????????, shape=[batch_size, max_ngram], [chunk_label: mention_mask ??????????????????????????????????????? 1, ???????????? 0]
            - chunk_mask:  ??? mention_mask ?????? -1-padding ????????????, mention_mask ?????? ??? ???????????????????????????????????????????????? flatten
        """
        # --------------------------------------------------------------------------------
        # Bert Embedding Outputs
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        sequence_output = outputs[0]  # bert ???????????????????????????????????? ?????????????????? ???????????? shape=[batch_size, max_sequence_length, hidden_size]

        # --------------------------------------------------------------------------------
        # Valid Outputs : get first token vector
        batch_size = sequence_output.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_ids[i]).item()                # ?????? ?????? ??? ?????????
            vectors = sequence_output[i][valid_ids[i] == 1]     # ????????? ??????????????? ?????? ??? ?????????
            valid_output[i, :valid_num].copy_(vectors)

        # --------------------------------------------------------------------------------
        # Dropout
        sequence_output = self.dropout(valid_output)

        # --------------------------------------------------------------------------------
        # CNN Outputs
        cnn_outputs = self.cnn2gram(sequence_output)  # shape = (batch_size, max_gram_num, 512)

        # --------------------------------------------------------------------------------
        # Classifier 512 to 1
        classifier_scores = self.classifier(cnn_outputs)  # shape = (batch_size, max_gram_num, 1)
        classifier_scores = classifier_scores.squeeze(-1)  # shape = (batch_size, max_gram_num),

        classifier_scores = classifier_scores.unsqueeze(1).expand(active_mask.size())  # shape = (batch_size, max_diff_ngram_num, max_gram_num)
        # activate_mask ??? ?????? 1 ?????????, classifier_scores ??????????????? ??????????????? value
        classifier_scores = classifier_scores.masked_fill(mask=active_mask, value=-float('inf'))
        # --------------------------------------------------------------------------------
        # Merge TF : # shape = (batch_size * max_diff_ngram_num * max_gram_num) to (batch_size * max_diff_ngram_num)
        total_scores, indices = torch.max(classifier_scores, dim=-1) # ???????????? ??????????????? ??? ???????????? ??? ?????? ??????????????????

        # --------------------------------------------------------------------------------
        # Total Loss Compute
        if labels is not None and chunk_labels is not None:
            # *************************************************************************************
            # *************************************************************************************
            # [1] Chunk Loss
            Chunk_Loss_Fct = CrossEntropyLoss(reduction='mean')

            active_chunk_loss = chunk_mask.view(-1) != -1      # chunk_mask ??? ??? padding ?????????
            chunk_logits = self.chunk_classifier(cnn_outputs)  # shape = (batch_size * num_gram, 2)
            active_chunk_logits = chunk_logits.view(-1, self.num_labels)[active_chunk_loss] # ??? ??? padding ??? logits, ????????????????????????????????????????????????????????????

            active_chunk_label_loss = chunk_labels.view(-1) != -1    # chunk_label ??? padding ?????????
            active_chunk_labels = chunk_labels.view(-1)[active_chunk_label_loss] # shape= [batch_size, max_ngram] ==> [batch_size * max_ngram]

            chunk_loss = Chunk_Loss_Fct(active_chunk_logits, active_chunk_labels)

            # *************************************************************************************
            # *************************************************************************************
            # [2] Rank Loss
            Rank_Loss_Fct = MarginRankingLoss(margin=0.5, reduction='mean')

            device = torch.device("cuda", total_scores.get_device())
            flag = torch.FloatTensor([1]).to(device)

            rank_losses = []
            for i in range(batch_size):
                score = total_scores[i]
                label = labels[i]

                true_score = score[label == 1]  # ???????????????, ?????????????????????   ????????? ??? ??????????????? ??? score
                neg_score = score[label == -1]  # ???????????????, ???????????? ???????????? ??? ??????????????? ??? score

                rank_losses.append(Rank_Loss_Fct(true_score.unsqueeze(-1), neg_score.unsqueeze(0), flag)) # ???????????????????????? score

            rank_loss = torch.mean(torch.stack(rank_losses))
            # *************************************************************************************
            # *************************************************************************************
            # [3] Total Loss
            tot_loss = rank_loss + chunk_loss
            return tot_loss

        else:
            return total_scores  # shape = (batch_size * max_differ_gram_num)
