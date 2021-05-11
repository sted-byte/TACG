import zcutils
import argparse
import os
import numpy as np
import copy
import transformers
import torch
import json

import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from sklearn import model_selection
from MCE import MCE
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import CrossEntropyLoss, HingeEmbeddingLoss
from matplotlib.font_manager import FontProperties
from scipy.interpolate import make_interp_spline

font = FontProperties(
    fname=r"/mnt/data/zc/anaconda3/envs/zc/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/simsun.ttc",
    size=14)
logger = None
tokenizer = None
max_length = None



def create_model(model_path):
    logger.info("start loading model...")
    # if pretrained_model_path is not None:
    #     logger.info("loading model from {}".format(pretrained_model_path))
    #     model_mt5 = MT5ForConditionalGeneration.from_pretrained(model_path)
    #     model = MCE(model_mt5, model_mt5.model_dim, args.classification_dim)
    # else:
    logger.info("loading model from {}".format(model_path))
    model_mt5 = torch.load(model_path,map_location=torch.device("cpu"))
    return model_mt5




def pre_process_test(data_test_path, tokenizer, max_length):
    result = []
    with open(data_test_path, "r", encoding="UTF-8") as f_in:
        lines = f_in.read()
    data_test = json.loads(lines)
    len_all = len(data_test)
    for i in range(len_all):
        temp = {}
        input_ids = tokenizer.prepare_seq2seq_batch(src_texts=[data_test[i]["desc"]], return_tensors="pt",
                                                    max_length=max_length).input_ids
        temp["desc"] = data_test[i]["desc"]
        temp["input_ids"] = input_ids
        temp["gt"] = data_test[i]["gt"]
        result.append(temp)
    return result

def pre_process_yago(data_path,tokenizer,n_ctx):
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
        desc=data[i]['infos']['itemListElement'][0]['result']['detailedDescription']['articleBody']
        name=data[i]['infos']['itemListElement'][0]['result']['name']
        if "@type" not in data[i]['infos']['itemListElement'][0]['result'].keys():
            continue
        gt=data[i]['infos']['itemListElement'][0]['result']['@type']
        # gt=data[i]['type']
        for j in range(len(gt)):
            temp={}
            temp["desc"]=desc # for test
            input_ids = tokenizer.prepare_seq2seq_batch(src_texts=[desc], return_tensors="pt",
                                                        max_length=max_length).input_ids
            temp["input_ids"]=input_ids     # for test
            temp["gt"]=gt             # for test
            temp["name"]=name
            temp["text"]=desc
            temp["label"]=gt[j]
            final_result.append(temp)
    return final_result


def pre_process(data_path):
    result_all=[]
    data = zcutils.load_pickle(data_path)
    result = data[:90000]
    result.extend(data[90999:])
    len_all = len(result)
    # len_all=100
    for i in tqdm(range(len_all)):
        desc = result[i]['cndbpedia'][0]
        sub_len = len(result[i]['cnprobase'])
        cepts = []
        for j in range(sub_len):
            cepts.append(result[i]['cnprobase'][j]['con'])
        for cept in cepts:
            tp={}
            tp["text"]=copy.deepcopy(desc)
            tp["label"]=cept
            result_all.append(tp)
    return result_all

def collate_fn(batch):
    result = {}
    batch_size = len(batch)
    src_texts = []
    src_tgts = []
    labels_classification = np.zeros(batch_size)
    gt_classification = [-1] * batch_size
    for i in range(batch_size):
        src_texts.append(batch[i]["text"])
    batch_t5 = tokenizer.prepare_seq2seq_batch(src_texts=src_texts, max_length=max_length, return_tensors="pt")
    input_ids = batch_t5.input_ids
    attention_mask = batch_t5.attention_mask
    for i in range(batch_size):
        src_tgts.append(batch[i]["label"])
    batch_t5 = tokenizer.prepare_seq2seq_batch(src_texts=src_tgts, max_length=max_length, return_tensors="pt")
    labels = batch_t5.input_ids
    # for i in range(batch_size):
    #     if batch[i]["task"] == 0:
    #         labels_classification[i] = 1
    #         gt_classification[i] = batch[i]["classification"]
    #         labels[i] = -100  # 不计算loss
    result["input_ids"] = input_ids
    result["attention_mask"] = attention_mask
    result["labels"] = labels
    result["labels_classification"] = labels_classification
    result["gt_classification"] = gt_classification
    return result


def train(model, train_list, device, args):
    train_dataset = zcutils.MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, drop_last=True, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)
    model.train()
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size)
    logger.info("We will process {} steps".format(total_steps))

    # 设置优化器
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    # 使用warmup策略
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=total_steps)
    # 分类器loss
    loss_fct = CrossEntropyLoss()

    logger.info("starting training.")
    overall_steps = 0
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, _ in enumerate(train_dataloader):
            input_ids = _["input_ids"]
            attention_mask = _["attention_mask"]
            labels = _["labels"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            loss= model.forward_t5pegasus(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                overall_steps += 1
            if (overall_steps + 1) % args.log_steps == 0:
                logger.info(
                    "batch {}/{} of epoch {}/{}, loss {}".format(batch_idx + 1, train_dataloader.__len__(), epoch + 1,
                                                                 args.epochs, loss.item(), ))
        if args.t5pegasus:
            model_path = os.path.join(args.saved_model_path_t5pegasus,"model_epoch{}.pt".format(epoch + 1))
        else:
            if args.mt5_small:
                model_path = os.path.join(args.saved_model_path_mt5_small, "model_epoch{}.pt".format(epoch + 1))
            else:
                model_path = os.path.join(args.saved_model_path_mt5_base, "model_epoch{}.pt".format(epoch + 1))
        logger.info("epoch {} finished".format(epoch + 1))
        if args.save_model:
            torch.save({"state_dict":model.state_dict(),
                        "args":args}, model_path)
        epoch_finish_time = datetime.now()
        logger.info("time for one epoch : {}".format(epoch_finish_time - epoch_start_time))
    logger.info("finished training")


def test(model, tokenizer, datalist, device, beam_size, out_test_path, is_mutual, distribution_validation):
    according_postive = True
    layer=8
    head=7
    result_all = []
    len_all = len(datalist)

    with torch.no_grad():
        for i in range(min(len_all,500)):
            input_ids = datalist[i]["input_ids"]
            input_ids = input_ids.to(device)
            temp = model.forward_test(input_ids, beam_size, is_mutual,
                                                    len(tokenizer) - 1)  # sep_id = len(tokenizer)-1
            if beam_size!=1:
                result=temp.sequences
                real_probability = np.array(torch.exp(temp.sequences_scores).cpu())
            else:
                result=temp


            preds = []
            for j in range(len(result)):
                pred = tokenizer.decode(result[j][1:-1]).split("<extra_id_0>")
                pred = [li.replace("<pad>", "") for li in pred]
                pred = [li.replace("</s>", "") for li in pred]
                pred = [li.strip() for li in pred]
                pred = list(set(pred))
                preds.extend(pred)
            # assert len(result)==1
            tp = {}
            tp["desc"] = datalist[i]["desc"]
            tp["gt"] = datalist[i]["gt"]
            tp["pred"] = preds

            tp["scores"]=real_probability
            print("desc:{}".format(tp["desc"]))
            real_preds=[]
            if beam_size!=1:
                for j in range(len(tp["pred"])):
                    print("{}:{:.5f}".format(tp["pred"][j], round(tp["scores"][j], 5)))
                pred_temp=[]
                scores_temp=[]
                for j in range(len(tp["pred"])):
                    if len(tp["pred"][j])<=15 and tp["scores"][j]>=0.1:
                        pred_temp.append(tp["pred"][j])
                        scores_temp.append(tp["scores"][j])
                tp["pred"]=pred_temp
                tp["scores"]=[float(_) for _ in scores_temp]

            else:
                for j in range(len(tp["pred"])):
                    print("{}".format(tp["pred"][j]))
            result_all.append(tp)

    if out_test_path is not None:
        result_all[i]["desc"]=result_all[i]["desc"].replace("\n","")
        with open(out_test_path, "w", encoding="UTF--8") as f_out:
            for i in range(len(result_all)):
                result_all[i]["desc"]=result_all[i]["desc"].replace("\n","")
                f_out.write("{}------{}------{}\n".format(result_all[i]["desc"],",".join(result_all[i]['gt']),",".join(result_all[i]["pred"])))
    logger.info("finished test")


def main(args):
    global logger, tokenizer, max_length
    logger = zcutils.create_logger(args.log_path)
    device = zcutils.device_info(args.device)
    max_length = args.max_length
    if args.t5pegasus:
        assert args.mt5_small == False
        from tokenizer import T5PegasusTokenizer
        tokenizer = T5PegasusTokenizer.from_pretrained(args.t5pegasus_tokenizer_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.mt5_path)

    if args.test:
        if args.t5pegasus:
            model_t5pegasus = create_model(args.pretrained_model_path_t5pegasus)
            model_temp = MT5ForConditionalGeneration.from_pretrained(args.t5pegasus_model_path)
            model=MCE(model_temp,model_temp.model_dim,args.classification_dim)
            model.load_state_dict(model_t5pegasus['state_dict'])
        else:
            if args.mt5_small:
                model_mt5 = create_model(args.pretrained_model_path_mt5_small)
                model_temp=MT5ForConditionalGeneration.from_pretrained(args.model_path_mt5_small)
            else:
                model_mt5 = create_model(args.pretrained_model_path_mt5_base)
                model_temp=MT5ForConditionalGeneration.from_pretrained(args.model_path_mt5_base)
            model = MCE(model_temp,model_temp.model_dim,args.classification_dim)
            model.load_state_dict(model_mt5['state_dict'])
    else:
        if args.t5pegasus:
            model = MT5ForConditionalGeneration.from_pretrained(args.t5pegasus_model_path)
            num_parameters = zcutils.model_paramters_num(model)
            logger.info("Model has {} parameters".format(num_parameters))
            model = MCE(model,model.model_dim,args.classification_dim)
            if not os.path.exists(args.saved_model_path_t5pegasus):
                logger.info("build folder {}".format(args.saved_model_path_t5pegasus))
                os.mkdir(args.saved_model_path_t5pegasus)
        else:
            if args.mt5_small:
                model = MT5ForConditionalGeneration.from_pretrained(args.model_path_mt5_small)
                if not os.path.exists(args.saved_model_path_mt5_small):
                    logger.info("build folder {}".format(args.saved_model_path_mt5_small))
                    os.mkdir(args.saved_model_path_mt5_small)
            else:
                model = MT5ForConditionalGeneration.from_pretrained(args.model_path_mt5_base)
                if not os.path.exists(args.saved_model_path_mt5_base):
                    logger.info("build folder {}".format(args.saved_model_path_mt5_base))
                    os.mkdir(args.saved_model_path_mt5_base)
            num_parameters = zcutils.model_paramters_num(model)
            logger.info("Model has {} parameters".format(num_parameters))
            model = MCE(model,model.model_dim,args.classification_dim)
    model.to(device)
    num_parameters = zcutils.model_paramters_num(model)
    logger.info("Model has {} parameters".format(num_parameters))
    if args.test:
        print("This is test")
        # datalist = pre_process_test(args.data_test_path, tokenizer, args.max_length)
        dataset = pre_process_yago(args.yago_data_path, tokenizer, args.max_length)
        train_list, test_list = model_selection.train_test_split(dataset, test_size=args.test_size, random_state=2021,
                                                                 shuffle=True)
        logger.info("we got {} test sample(s)".format(len(test_list)))
        model.eval()
        test(model, tokenizer, test_list, device, args.beam_size, args.out_test_path, args.is_mutual,
             args.distribution_validation)
    else:
        dataset = pre_process_yago(args.yago_data_path, tokenizer, args.max_length)
        train_list, test_list = model_selection.train_test_split(dataset, test_size=args.test_size, random_state=2021,
                                                                 shuffle=True)
        train(model, train_list, device, args)
    print("END _ MAIN")
    print("end")

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="mt5 for concepts generation with mutual concepts")
    parse.add_argument("--cn_po_ce_path",type=str,default="raw_data/cn_po_info.pkl")
    parse.add_argument("--cn_po_mu_ce_path", type=str, default="raw_data/cn_po_mu_ce_info.pkl",
                       help="source of raw data")
    parse.add_argument("--log_path", type=str, default="log.txt", help="path of log file")
    parse.add_argument("--device", type=str, default="gpu", help="what device you want to use,can be gpu or cpu")
    parse.add_argument("--mt5_path", type=str, default="/home/zc/projects/hfl/mt5-base", help="path of mt5 folder")
    parse.add_argument("--test", type=bool, default=True, help="if test is False,need to give pretrained_model_path")
    parse.add_argument("--pretrained_model_path_t5pegasus", type=str, default="saved_model_t5_pegasus/model_epoch5.pt")
    parse.add_argument("--pretrained_model_path_mt5_small",type=str, default="saved_model_mt5_small/model_epoch5.pt")
    parse.add_argument("--pretrained_model_path_mt5_base",type=str,default="saved_model_mt5_base_freebase/model_epoch5.pt")
    parse.add_argument("--model_path_t5pegasus",type=str,default="t5_pegasus_torch")
    parse.add_argument("--model_path_mt5_base",type=str,default="/home/zc/projects/hfl/mt5-base")
    parse.add_argument("--model_path_mt5_small",type=str,default="/mnt/data/zc/projects/hfl/mt5-small")
    parse.add_argument("--saved_model_path_t5pegasus", type=str, default="saved_model_t5_pegasus")
    parse.add_argument("--saved_model_path_mt5_small",type=str,default="saved_model_mt5_small")
    parse.add_argument("--saved_model_path_mt5_base",type=str,default="saved_model_mt5_base_freebase")
    parse.add_argument("--max_length", type=int, default=128, help="The max length of sentence")
    parse.add_argument("--bernoulli", type=float, default=0, help="The bernoulli ratio for 1 and 0")
    parse.add_argument("--classification_task_rate", type=float, default=1, help="rate of classification task")
    parse.add_argument("--test_size", type=float, default=0.1, help="The ratio of test size")
    parse.add_argument("--classification_dim", type=int, default=1, help="The linear classification dim")
    parse.add_argument("--num_workers", type=int, default=1, help="number of workers of DataLoader")
    parse.add_argument("--batch_size", type=int, default=2, help="batch size")
    parse.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parse.add_argument("--lr", type=float, default=1.5e-4, help="learning rate")
    parse.add_argument("--warmup_steps", type=int, default=4000, help="number of warmup steps")
    parse.add_argument("--max_grad_norm", type=float, default=1.0, help="Max-Norm Regularization ")
    parse.add_argument("--gradient_accumulation", type=int, default=1, help="number of gradient accumulation steps")
    parse.add_argument("--log_steps", type=int, default=1, help="number of log steps")
    parse.add_argument("--save_model", type=bool, default=True, help="whether save model")
    parse.add_argument("--train_mode", type=str, default="together",
                       help="concepts together or separated, can be together or separated")
    parse.add_argument("--beam_size", type=int, default=8, help="size of beam search")
    parse.add_argument("--data_test_path", type=str, default="raw_data/gt_999.txt", help="path of test input file")
    parse.add_argument("--out_test_path", type=str, default="result_mt5_base_freebase_pure.txt",
                       help="path of test output file")
    parse.add_argument("--t5pegasus",type=bool,default=False, help="whether to use t5pegasus")
    parse.add_argument("--mt5_small",type=bool,default=False)
    parse.add_argument("--t5pegasus_model_path",type=str,default="t5_pegasus_torch")
    parse.add_argument("--t5pegasus_tokenizer_path",type=str,default="t5_pegasus_torch")
    parse.add_argument("--yago_data_path",type=str,default="yago_freebase.json")
    args = parse.parse_args()
    main(args)
