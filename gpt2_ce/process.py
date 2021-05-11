import numpy as np
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Model
import transformers
import logging
import CONFIG
import os
import sklearn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
from zc import zcutils

logger = None
pad_id = 0
PAD = '[PAD]'


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def create_logger(log_path):
    """
       将日志输出到日志文件和控制台
       """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def create_model(vocab_size, model_config_path, pretrained_model_path=None):
    if pretrained_model_path is not None:
        model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)
    else:
        model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(model_config_path)
        model = GPT2LMHeadModel(config=model_config)
    model.resize_token_embeddings(vocab_size)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get('n_ctx')


def pre_process_test(data_test_path, tokenizer, n_ctx, cept_path, num_classification):
    result_data = []
    with open(data_test_path, "r", encoding="UTF-8") as f_in:
        data_list = f_in.readlines()
        for i in range(len(data_list)):
            temp = data_list[i]
            desc = temp.strip("\n").split("\t")[1]
            cepts = temp.strip("\n").split("\t")[2].split(",")
            dic = {}
            dic["index"] = i
            dic["text"] = desc[n_ctx-20]
            dic["gt"] = cepts
            result_data.append(dic)
    len_all = len(result_data)
    for i in tqdm(range(len_all)):
        text_ids = [tokenizer.cls_token_id]
        # text_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in result_data[i]["text"]])
        # text_ids = text_ids[:n_ctx - 1]
        text_ids.extend(tokenizer.encode(result_data[i]['text'])[1:-1])
        text_ids.append(1)
        result_data[i]["ids"] = text_ids
    return result_data



def pre_process_testall(data_testall_path, tokenizer, n_ctx,cfg):
    result_data = []
    with open(data_testall_path, "r", encoding="UTF-8") as f_in:
        data_list = f_in.readlines()
        for i in range(len(data_list)):
            temp = data_list[i]
            desc = temp.strip("\n").split("\t")[0]
            topic = temp.strip("\n").split("\t")[1]
            dic = {}
            dic["index"] = i
            dic["text"] = desc
            dic["topic"] = topic
            result_data.append(dic)
    len_all = len(result_data)
    for i in tqdm(range(len_all)):
        text_ids = [tokenizer.cls_token_id]
        # text_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in result_data[i]["text"]])
        if cfg.together:
            text_ids.extend(tokenizer.encode(result_data[i]["text"])[1:-1])
        else:
            text_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in result_data[i]["text"]])
        text_ids = text_ids[:n_ctx - 1]
        if cfg.together:
            text_ids.extend([1])
        else:
            text_ids.extend([tokenizer.sep_token_id])
        result_data[i]["ids"] = text_ids
    return result_data

def pre_process_test_standard(data_path,tokenizer,n_ctx,cfg):
    data = zcutils.load_pickle(data_path)
    result = data[90000:90999]
    len_all = len(result)
    result_data = []
    for i in range(len_all):
        dic={}
        dic["index"]=i
        desc=result[i]["cndbpedia"][0]
        sub_len = len(result[i]["cnprobase"])
        topic=[]
        for j in range(sub_len):
            topic.append(result[i]["cnprobase"][j]["con"])
        dic["topic"]=topic
        dic["text"]=desc[:n_ctx-20]
        result_data.append(dic)
    len_all=len(result_data)
    for i in tqdm(range(len_all)):
        text_ids = [tokenizer.cls_token_id]
        if cfg.together:
            text_ids.extend(tokenizer.encode(result_data[i]["text"])[1:-1])
        else:
            text_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in result_data[i]["text"]])
        text_ids = text_ids[:n_ctx - 1]
        if cfg.together:
            text_ids.extend([1])
        else:
            text_ids.extend([tokenizer.sep_token_id])
        result_data[i]["ids"] = text_ids
    return result_data

def pre_process_yago_test(data_path,tokenizer,n_ctx):
    with open(data_path,"r",encoding="UTF-8") as f_in:
        data=json.loads(f_in.read())
    final_result=[]
    len_all=len(data)
    # len_all=500
    for i in tqdm(range(len_all)):
        if len(data[i]['infos']['itemListElement'])==0:
            continue
        # print(data[i]['infos']['itemListElement'])
        if "detailedDescription" not in data[i]['infos']['itemListElement'][0]['result'].keys():
            continue
        desc=data[i]['infos']['itemListElement'][0]['result']['detailedDescription']['articleBody'][:n_ctx-20]
        name=data[i]['infos']['itemListElement'][0]['result']['name']
        if "@type" not in data[i]['infos']['itemListElement'][0]['result'].keys():
            continue
        gt=data[i]['infos']['itemListElement'][0]['result']['@type']
        text_ids=tokenizer.encode(desc)
        temp={}
        temp["text"]=desc
        temp["ids"]=text_ids
        temp["topic"]=gt
        temp["index"]=i
        final_result.append(temp)

    return final_result

def clear_data(now):
    print("准备清理{}条数据".format(len(now)))
    len_all = len(now)
    peo = 0
    plant = 0
    for i in range(len_all):
        if i >= 90000 and i < 90999:
            continue
        sub_len = len(now[i]["cnprobase"])
        temp_cnprobase = []
        cepts = []
        for j in range(sub_len):
            cepts.append(now[i]["cnprobase"][j]['con'])
        if "人物" in cepts or "官员" in cepts or "大臣" in cepts:
            finding = False
            for j in range(sub_len):
                if cepts[j] != "历史" and cepts[j] != "历史书籍":
                    temp_cnprobase.append(now[i]["cnprobase"][j])
                else:
                    # print(now[i])
                    finding = True
            if finding:
                peo += 1
            now[i]["cnprobase"] = temp_cnprobase
        elif "植物" in cepts or "生物" in cepts:
            finding = False
            for j in range(sub_len):
                if cepts[j] != "中医" and cepts[j] != "医生":
                    temp_cnprobase.append(now[i]["cnprobase"][j])
                else:
                    # print(now[i])
                    finding = True
            plant += 1
            now[i]["cnprobase"] = temp_cnprobase
    print("清洗出{}条人物错误数据\n清洗出{}条植物错误数据".format(peo, plant))
    return now


def pre_process(data_path, tokenizer, n_ctx, cept_path, num_classification):
    data = zcutils.load_pickle(data_path)
    logger.info("开始清洗数据")
    # data = clear_data(data)
    logger.info("结束清洗数据")
    # zcutils.save_pickle(data, "cn_po_info_clear.pkl")
    result = data[0:90000]
    result.extend(data[90999:])
    len_all = len(result)
    # len_all = 300
    final_result = []
    all_result = []
    for i in range(len_all):
        sub_len = len(result[i]["cnprobase"])
        text = result[i]["cndbpedia"][0]
        tgt = []
        for j in range(sub_len):
            tgt.append(result[i]["cnprobase"][j]["con"])
        result[i]["text"] = text[:n_ctx - 20]
        result[i]["tgt"] = tgt
        all_result.append(result[i])

    # token to ids
    result = all_result
    len_all = len(result)
    for i in tqdm(range(len_all)):
        text_ids = [tokenizer.cls_token_id]
        text_ids.extend(tokenizer.encode(result[i]["text"])[1:-1])

        temp=deepcopy(text_ids)
        for j in range(len(result[i]["tgt"])):
            temp.append(1)
            temp.extend(tokenizer.encode(result[i]["tgt"][j])[1:-1])
        temp.append(tokenizer.sep_token_id)
        tp=deepcopy(result[i])
        tp['text_ids']=temp
        tp['all_tgt_ids']=temp
        final_result.append(tp)

    return final_result

def pre_process_probase(data_path,tokenizer,n_ctx,cfg):
    with open(data_path,"r",encoding='UTF-8') as f_in:
        data=json.loads(f_in.read())
    final_result=[]
    len_all=len(data)
    for i in tqdm(range(len_all)):
        text_ids=tokenizer.encode(data[i]['desc'][:n_ctx-20])[:-1]
        for j in range(len(data[i]["gt"])):
            temp={}
            temp["text_ids"]=deepcopy(text_ids)
            temp["text_ids"].append(tokenizer.sep_token_id)
            temp["text_ids"].extend(tokenizer.encode(data[i]["gt"][j])[1:])
            temp["name"]=data[i]["name"]
            temp["desc"]=data[i]["desc"]
            temp["gt"]=data[i]["gt"]
            final_result.append(temp)
    return final_result

def pre_process_yago(data_path,tokenizer,n_ctx):
    with open(data_path,"r",encoding="UTF-8") as f_in:
        data=json.loads(f_in.read())
    final_result=[]
    len_all=len(data)
    # len_all=500
    for i in tqdm(range(len_all)):
        if len(data[i]['infos']['itemListElement'])==0:
            continue
        if "detailedDescription" not in data[i]['infos']['itemListElement'][0]['result'].keys():
            continue
        desc=data[i]['infos']['itemListElement'][0]['result']['detailedDescription']['articleBody'][:n_ctx-20]
        name=data[i]['infos']['itemListElement'][0]['result']['name']
        if "@type" not in data[i]['infos']['itemListElement'][0]['result'].keys():
            continue
        gt=data[i]['infos']['itemListElement'][0]['result']['@type']

        text_ids=tokenizer.encode(desc)[:-1]
        for j in range(len(gt)):
            temp={}
            temp["text_ids"]=deepcopy(text_ids)
            temp["text_ids"].append(tokenizer.sep_token_id)
            temp["text_ids"].extend(tokenizer.encode(gt[j])[1:])
            temp["name"]=name
            temp["desc"]=desc
            temp["gt"]=gt
            final_result.append(temp)
    return final_result




def calculate_loss_and_accuracy(outputs, labels, device, tokenizer):
    logits = outputs[0]
    # print(outputs[0].shape)
    len_all = len(labels)
    shift_logits = logits[..., :-1, :].contiguous()
    all_loss = torch.zeros(1, dtype=torch.float32).to(device)
    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    all_num_targets = 0
    corrects = 0.0
    for i in range(len_all):
        min_loss = torch.FloatTensor([1e9]).to(device)
        real_num_targets = 0
        k = -1
        _, ind_max = torch.max(shift_logits[i], 1)
        # print("Pred result is {}".format("".join(tokenizer.convert_ids_to_tokens(tp.item()) for tp in ind_max)))
        for j in range(len(labels[i])):
            shift_labels = labels[i][j][1:].contiguous().to(device)
            # print("The {} shift_lables is {}".format(j,"".join(tokenizer.convert_ids_to_tokens(tp.item()) for tp in shift_labels)))
            tp_loss = loss_fct(shift_logits[i], shift_labels)
            not_ignore = shift_labels.ne(pad_id)
            num_targets = not_ignore.long().sum().item()
            # print("This loss is {}/{}={}".format(tp_loss.item(),num_targets,tp_loss.item()/num_targets))
            if tp_loss / num_targets < min_loss:
                min_loss = tp_loss
                real_num_targets = num_targets
                k = j
        # print("minimum loss is {}".format(min_loss.item()/real_num_targets))
        _, preds = shift_logits[i].max(dim=-1)
        shift_labels = labels[i][k][1:].contiguous().to(device)
        not_ignore = shift_labels.ne(pad_id)
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        corrects += correct
        all_num_targets += real_num_targets
        all_loss += min_loss
    accuracy = corrects / all_num_targets
    # print("loss: {}, accuracy: {}".format(all_loss.item(),accuracy))
    return all_loss / all_num_targets, accuracy


def calculate_loss_and_accuracy_stable(outputs, labels, device, tokenizer):
    logits = outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def cmp(temp):
    return temp[1]


def test(model, tokenizer, test_list, device, cfg):
    with open(cfg.saved_out_path, "w", encoding="UTF-8") as f_out:
        with torch.no_grad():
            for i in range(0, min(500,len(test_list))):
                curr_input_tensor = torch.tensor(test_list[i]["ids"], dtype=torch.long).to(device)
                desc = test_list[i]["text"][:cfg.n_ctx - 21]
                print("desc:{}".format(desc))
                dic = {}
                dic["desc"] = desc
                dic["preds"] = []
                dic["scores"] = []
                dic["index"] = test_list[i]["index"]
                dic["topic"] = test_list[i]["topic"]

                outputs = model.generate(input_ids=curr_input_tensor.unsqueeze(0), num_beams=cfg.topk,
                                         num_return_sequences=cfg.topk, eos_token_id=tokenizer.sep_token_id,
                                         max_length=cfg.n_ctx + cfg.test_max_len, output_scores=True,
                                         return_dict_in_generate=True)
                print("{}".format(desc))
                if cfg.together:
                    temp=list(outputs[0][0])
                    now=[]
                    for j in range(len(temp)):
                        if temp[j]==1 or temp[j]==102:
                            now.append(j)
                    for j in range(len(now)-1):
                        pred="".join([tokenizer.decode(_) for _ in temp[now[j]+1:now[j+1]]])
                        dic["preds"].append(pred)
                    print(dic["preds"])
                else:
                    for j in range(cfg.topk):
                        now = []
                        temp = list(outputs[0][j])
                        for k in range(len(temp)):
                            if temp[k] == 102:
                                now.append(k)
                        assert len(now) == 2
                        pred = tokenizer.decode(outputs[0][j][now[0] + 1:now[1]])
                        score = torch.exp(outputs[1][j] * (list(outputs[0][j]).index(102) + 1))
                        if score > 0.1 or pred in desc:
                            print("Concept:{} Score:{:.5f}".format(pred, score))
                            dic["preds"].append(pred)
                desc=desc.replace("\n","")
                f_out.write("{}------{}------{}\n".format(desc, ",".join(dic["topic"]), ",".join(dic["preds"])))
                # for j in range(len(concepts)):
                #     pred="".join([tokenizer.convert_ids_to_tokens(_.item()) for _ in concepts[j][0][:-1]])
                #     score=concepts[j][1]
                #     dic["preds"].append(pred)
                #     dic["scores"].append(score.item())
                #     print("Concept{}:{},score:{:.3f}".format(j+1,pred,score.item()))
                del curr_input_tensor

def test_probase(model, tokenizer, test_list, device, cfg):
    name_set=set()
    all_test=[]
    len_all=len(test_list)
    for i in range(len_all):
        if test_list[i]["name"] not in name_set:
            all_test.append(test_list[i])
            name_set.add(test_list[i]["name"])
    len_all=len(all_test)
    len_all=500
    logger.info("we got {} test samples.".format(len_all))
    with open(cfg.saved_out_path,"w",encoding="UTF-8") as f_out:
        for i in tqdm(range(len_all)):
            desc=all_test[i]["desc"]
            curr_input_tensor=torch.tensor(tokenizer.encode(desc[:cfg.n_ctx-20]), dtype=torch.long).to(device)
            print("{}".format(desc[:cfg.n_ctx-20]))
            outputs = model.generate(input_ids=curr_input_tensor.unsqueeze(0), num_beams=cfg.topk,
                                     num_return_sequences=cfg.topk, eos_token_id=tokenizer.sep_token_id,
                                     max_length=cfg.n_ctx + cfg.test_max_len, output_scores=True,
                                     return_dict_in_generate=True)
            dic={}
            dic["desc"]=desc
            dic["preds"]=[]
            dic["gt"]=all_test[i]["gt"]
            for j in range(cfg.topk):
                now = []
                temp = list(outputs[0][j])
                for k in range(len(temp)):
                    if temp[k] == 102:
                        now.append(k)
                assert len(now) == 2
                pred = tokenizer.decode(outputs[0][j][now[0]+1:now[1]])
                score = torch.exp(outputs[1][j] * (now[1]-now[0]))
                print("Concept:{} Score:{:.5f}".format(pred, score))
                if score > 0.1 or pred in desc:
                    # print("Concept:{} Score:{:.5f}".format(pred, score))
                    dic["preds"].append(pred)
            f_out.write("{}------{}------{}\n".format(dic["desc"].replace("\n","")[:cfg.n_ctx-20],",".join(dic["gt"]),",".join(dic["preds"])))




def train(model, train_list, device, cfg, tokenizer):
    train_dataset = zcutils.MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, drop_last=True, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, collate_fn=zcutils.collate_fn)
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))

    # 设置优化器
    optimizer = transformers.AdamW(model.parameters(), lr=cfg.lr, correct_bias=True)
    # 使用warmup策略
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps,
                                                             num_training_steps=total_steps)

    logger.info("starting training.")
    overall_step = 0
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, _ in enumerate(train_dataloader):
            batch_size = len(_)
            sub_len = len(_[0]["text_ids"])
            input_ids = torch.zeros(batch_size, sub_len, dtype=torch.long)
            # tgt_ids = []
            for i in range(batch_size):
                input_ids[i] = deepcopy(_[i]["text_ids"])
                # tgt_ids.append(deepcopy(_[i]["all_tgt_ids"]))
            input_ids = input_ids.to(device)
            outputs = model.forward(input_ids)
            # loss,accuracy=calculate_loss_and_accuracy(outputs,labels=tgt_ids,device=device,tokenizer=tokenizer)
            loss, accuracy = calculate_loss_and_accuracy_stable(outputs, labels=input_ids, device=device,
                                                                tokenizer=tokenizer)
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % cfg.gradient_accumulation == 0:
                # 更新参数
                optimizer.step()
                # 清空梯度信息
                optimizer.zero_grad()
                # 进行warm up
                scheduler.step()
                overall_step += 1
            if (overall_step + 1) % cfg.log_step == 0:
                logger.info(
                    "batch {}/{} of epoch {}/{}, loss {}, accuracy {}".format(batch_idx + 1, train_dataloader.__len__(),
                                                                              epoch + 1, cfg.epochs, loss.item(),
                                                                              accuracy))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))

        logger.info("epoch {} finished.".format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if cfg.save_mode:
            model.save_pretrained(model_path)
        epoch_finish_time = datetime.now()
        logger.info("time for one epoch : {}".format(epoch_finish_time - epoch_start_time))
    logger.info("finished train")


def main():
    global logger
    cfg = CONFIG.CONFIG()
    device = zcutils.device_info(cfg.device)
    print(device)
    logger = create_logger(cfg.log_path)
    tokenizer = transformers.BertTokenizer.from_pretrained(cfg.tokenizer_path)
    vocab_size = len(tokenizer)

    if cfg.test:
        model, n_ctx = create_model(vocab_size, cfg.model_config_path, cfg.pretrained_model_path)
    else:
        model, n_ctx = create_model(vocab_size, cfg.wiki_pretrained_model_path, cfg.wiki_pretrained_model_path)
    model.to(device)
    print(n_ctx)
    n_ctx = cfg.n_ctx
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    zcutils.pad_id = pad_id
    if not os.path.exists(cfg.saved_model_path):
        logger.info("build mkdir {}".format(cfg.saved_model_path))
        os.mkdir(cfg.saved_model_path)
    # 获得模型参数个数
    num_parameters = zcutils.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))

    if cfg.test:
        print("This is test.")
        if cfg.probase:
            dataset = pre_process_probase(cfg.probase_data_path, tokenizer, n_ctx,cfg)
            train_list, test_list = model_selection.train_test_split(dataset, test_size=cfg.test_size,random_state=2021,shuffle=True)

            logger.info("we got {} test sample(s).".format(len(test_list)))



            result=[]
            nameset=set()
            for i in range(len(train_list)):
                if train_list[i]["name"] not in nameset:
                    nameset.add(train_list[i]["name"])
                    dic={}
                    dic["name"]=train_list[i]["name"]
                    dic["gt"]=train_list[i]["gt"]
                    dic["desc"]=train_list[i]["desc"]
                    result.append(dic)
            print("训练用例一共{}个".format(len(result)))
            with open("probase_train.txt","w",encoding="UTF-8") as f_out:
                f_out.write("{}\n".format(json.dumps(result,ensure_ascii=False,indent=2)))


            model.eval()
            test_probase(model, tokenizer, test_list, device, cfg)
        elif cfg.yago:
            dataset = pre_process_yago_test(cfg.yago_data_path, tokenizer, n_ctx)
            train_list, test_list = model_selection.train_test_split(dataset, test_size=cfg.test_size,random_state=2021, shuffle=True)
            test(model,tokenizer,test_list,device,cfg)
        else:
            dataset=pre_process_test_standard(cfg.data_path,tokenizer,n_ctx)
            train_list, test_list = model_selection.train_test_split(dataset, test_size=cfg.test_size,random_state=2021,shuffle=True)
            test(model,tokenizer,test_list,device,cfg)
    else:
        if cfg.probase:
            dataset=pre_process_probase(cfg.probase_data_path,tokenizer,n_ctx)
            train_list, test_list = model_selection.train_test_split(dataset, test_size=cfg.test_size,random_state=2021,shuffle=True)

            train(model, train_list, device, cfg, tokenizer)
        elif cfg.yago:
            dataset=pre_process_yago(cfg.yago_data_path,tokenizer,n_ctx)
            train_list,test_list=model_selection.train_test_split(dataset,test_size=cfg.test_size,random_state=2021,shuffle=True)
            train(model,train_list,device,cfg,tokenizer)
        else:
            dataset = pre_process(cfg.data_path, tokenizer, n_ctx, cfg.cept_path, cfg.num_classification)
            logger.info("we got {} sample(s).".format(len(dataset)))
            train(model, dataset, device, cfg, tokenizer)
        # train_list, test_list = model_selection.train_test_split(dataset, test_size=cfg.test_size)
        #         # logger.info("we got {} train sample(s) and {} test sample(s)".format(len(train_list), len(test_list)))
        #
        #         # print(len(dataset))
        #         # for i in range(10):
        #         #     print(dataset[i]["text"])
        #         #     print(dataset[i]["tgt"])

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    print("hll")
    main()
