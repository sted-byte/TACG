

class CONFIG:
    def __init__(self):
        self.log_path = "log.txt"
        self.tokenizer_path = "/home/zc/projects/hfl/chinese-roberta-wwm-ext"
        self.model_config_path = "config/model_config.json"
        self.device =   "gpu"
        self.saved_model_path = "saved_model_freebase"
        self.pretrained_model_path = "saved_model_2_5/model_epoch14"
        self.pretrained_model_path = "saved_model_freebase/model_epoch20"
        self.wiki_pretrained_model_path="GPT2_wiki"
        self.test = True
        self.num_workers = 1  # 使用Data
        self.data_path = "process_data/cn_po_info.pkl"
        self.tail="music"
        self.data_test_path="./3_11/{}.txt".format(self.tail)
        self.test_size = 0.1
        self.batch_size=16
        self.epochs=100
        self.lr=1.5e-4
        self.warmup_steps=2000
        self.log_step=1
        self.max_grad_norm=1.0
        self.gradient_accumulation=1
        self.topk=8
        self.test_max_len=20
        self.save_mode=True
        self.save_out_path="finetune_{}_top{}_{}.txt".format(self.pretrained_model_path.split("/")[-1],self.topk,self.tail)
        self.n_ctx=256
        self.cept_path="/mnt/data/zc/projects/nlg/gpt2_ce/process_data/raw_data/ce_200.txt"
        self.num_classification=19
        self.saved_out_path="result_freebase.txt"
        self.data_testall_path="testall.txt"
        self.together=False
        self.probase=False
        self.probase_data_path="process_data/probase_data.txt"
        self.yago=True
        self.yago_data_path="process_data/yago_freebase.json"