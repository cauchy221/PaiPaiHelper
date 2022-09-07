import numpy as np
import unicodedata, re
import torch
import torch.nn.functional as F


def get_word_weight(weightfile="", weightpara=2.7e-4):
    """
    Get the weight of words by word_fre/sum_fre_words
    :param weightfile
    :param weightpara
    :return: word2weight[word]=weight : a dict of word weight
    """
    if weightpara <= 0:  # when the parameter makes no sense, use unweighted
        weightpara = 1.0
    word2weight = {}
    word2fre = {}
    with open(weightfile, encoding='UTF-8') as f:
        lines = f.readlines()
    # sum_num_words = 0
    sum_fre_words = 0
    for line in lines:
        word_fre = line.split()
        # sum_num_words += 1
        if (len(word_fre) >= 2):
            word2fre[word_fre[0]] = float(word_fre[1])
            sum_fre_words += float(word_fre[1])
        else:
            print(line)
    for key, value in word2fre.items():
        word2weight[key] = weightpara / (weightpara + value / sum_fre_words)
        # word2weight[key] = 1.0 #method of RVA
    return word2weight


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    """

    Parameters
    ----------
    model: 编码模型
    input_ids: (b, l)
    attention_mask: (b, l)
    start_tokens: 对bert而言就是[101]
    end_tokens: [102]

    Returns
    -------

    """
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)   # 转化为tensor放在指定卡上
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)   # 1
    len_end = end_tokens.size(0)       # 1 if bert , 2 if roberta
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []   # num_seg记录原来的样本被切成多少片，1 or 2
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()  # 在len维度上求和，即每个样本的1的个数，即长度
        for i, l_i in enumerate(seq_len):
            # 对batch中的每一个样本循环
            if l_i <= 512:
                # 如果长度小于512，就直接添加
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                # 超过512的样本
                # 第一段取开始到511，加结束符
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                # 第二段取开始符，加剩下的部分
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                # attention_mask同理
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        # 在batch维度上拼接
        # 原本的input_ids 是(b, l)，经过上面的for循环new_input_ids每一项是(l,)
        # 然后在dim=0上stack，变回了(b, l)
        # 但是此时的b可能已经大于原来的batch_size
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # 把新构建的输入进行建模，然后把建模结果拼回原来的
        sequence_output = output[0]   # (b, l, 768)
        attention = output[-1][-1]    # (b, ?, l, l)
        i = 0   # i是旧的batch号
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                # 这个pad没看懂。n_s == 1的话，c - 512应该小于0
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                # 取第一个片段的建模结果
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]  # 构建第一个样本的时候增加了结束符，所以要去掉它
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                # 第二个片段的建模结果
                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]   # 构建第二个样本的时候增加了开始符，所以要从1开始索引，去掉它
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])

                # 把两个片段合并
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention


def rematch(text, tokens, do_lower_case=True):
    if do_lower_case:
        text = text.lower()
        
    def is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')
    
    def is_special(ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')
    
    def stem(token):
        if token[:2] == '##':
            return token[2:]
        else:
            return token
        
    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if do_lower_case:
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'mn'])
        ch = ''.join([c for c in ch if not (ord(c) == 0 or ord(c) == 0xfffd or is_control(c))])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        if token.startswith('▁'):
            token = token[1:]
        if is_special(token):
            token_mapping.append([])
        else:
            token = stem(token)
            if do_lower_case:
                token = token.lower()
            try:
                start = text[offset:].index(token) + offset
            except Exception as e:
                print(e)
                print(token)
            end = start + len(token)
            token_mapping.append(char_mapping[start: end])
            offset = end
            
    return token_mapping
