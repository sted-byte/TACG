import torch
from pymongo import MongoClient
import re
import pickle
from tqdm import tqdm
from copy import deepcopy

pad_id=None
# 设置text_len_flexible = True 使得 collate_fn 的每个batch自动填充
text_len_flexible=True
text_len_stable=50

def device_info(device):
    result = "cpu"
    if torch.cuda.is_available():
        counter = torch.cuda.device_count()
        print("There are {} GPU(s) is available.".format(counter))
        for i in range(counter):
            print("GPU {} Name:{}".format(i, torch.cuda.get_device_name(i)))
        if device == "gpu":
            result = "cuda:0"
            print("We will use {}".format(result))
    return result


def model_paramters_num(model):
    result = 0
    parameters = model.parameters()
    for paramter in parameters:
        result += paramter.numel()
    return result


# 加载和保存pickle文件

def load_pickle(path):
    with open(path, 'rb') as fil:
        data = pickle.load(fil)
    return data


def save_pickle(en, path):
    with open(path, 'wb') as fil:
        pickle.dump(en, fil)


def collate_fn(batch):
    global pad_id
    assert pad_id is not None
    input_ids=[]
    batch_size=len(batch)
    max_input_len=0
    # for i in range(batch_size):
    #     max_input_len=max(max_input_len,len(batch[i]))

    # for i in range(batch_size):
    #     for j in range(len(batch[i]["all_tgt_ids"])):
    #         max_input_len=max(max_input_len,len(batch[i]["all_tgt_ids"][j]))


    # this
    # for i in range(batch_size):
    #     max_input_len=max(max_input_len,len(batch[i]["all_tgt_ids"]))

    for i in range(batch_size):
        max_input_len=max(max_input_len,len(batch[i]["text_ids"]))

    # for i in range(batch_size):
    #     now_len=len(batch[i])
    #     input_ids.append(batch[i])
    #     input_ids[i].extend([pad_id]*(max_input_len-now_len))
    for i in range(batch_size):
        now_len = len(batch[i]["text_ids"])
        batch[i]["text_ids"].extend([pad_id] * (max_input_len - now_len))
        batch[i]["text_ids"]=torch.tensor(batch[i]["text_ids"],dtype=torch.long)
        # for j in range(len(batch[i]["all_tgt_ids"])):
        #     now_len=len(batch[i]["all_tgt_ids"][j])
        #     batch[i]["all_tgt_ids"][j].extend([pad_id]*(max_input_len-now_len))
        #     batch[i]["all_tgt_ids"][j]=torch.tensor(batch[i]["all_tgt_ids"][j],dtype=torch.long)
    # return torch.tensor(input_ids,dtype=torch.long)
    return batch

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits



class MyDataset:
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return deepcopy(self.data_list[item])



