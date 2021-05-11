# TACG
Code for the paper "A Novel Framework for Topic-Alignment Concept Generation"

## Requirements

- transformers>=4.2.0
- python version>=3.6
- [pytorch](http://pytorch.org/) version>=1.2.0

## Dataset

sample dataset is release on raw_data/

## Pre-trained text generation model for concept generation

- gpt2_ce

  You can adjust the hyperparameter in the gpt2_ce/CONFIG.py file

  ```
  python gpt2_ce.process.py 
  ```

- mt5_ce and mt5_pegasus_ce

  All the hyperparameters have default values and can be adjusted in the mt5_ce/process.py file, including MT5 and MT5_pegasus

  For example

  ```
  python mt5_ce.process.py \  
  --cn_po_ce_path raw_data/cn_po_info.pkl \  
  --log_path log.txt \  
  --device gpu \  
  --mt5_path /home/zc/projects/hfl/mt5-base \  
  --pretrained_model_path_t5pegasus saved_model_t5_pegasus/model_epoch5.pt \  
  --pretrained_model_path_mt5_small saved_model_mt5_small/model_epoch5.pt \  
  --pretrained_model_path_mt5_base saved_model_mt5_base_freebase/model_epoch5.pt \  
  --model_path_t5pegasus t5_pegasus_torch \  
  --model_path_mt5_base /home/zc/projects/hfl/mt5-base \  
  --model_path_mt5_small /mnt/data/zc/projects/hfl/mt5-small \  
  --saved_model_path_t5pegasus saved_model_t5_pegasus \  
  --saved_model_path_mt5_small saved_model_mt5_small \  
  --saved_model_path_mt5_base saved_model_mt5_base_freebase \ 
  --max_length 128 \  
  --test_size 0.1 \  
  --num_workers 1 \  
  --batch_size 8 \  
  --epochs 20 \  
  --lr 1.5e-4 \  
  --warmup_steps 4000 \  
  --max_grad_norm 1.0 \  
  --gradient_accumulation 1 \  
  --log_steps 1 \  
  --beam_size 8 \  
  --out_test_path test_out.txt \  
  --yago_data_path yago_freebase.json  
  ```
## text-topic classifier
### Data construction
The data should be constructed as <text>, for example
```
Vitaliy Fedoriv is a Ukrainian football defender who plays for Metal Kharkiv.
```
Then we run the text-topic classifier based on BERT
 ```
 python run_classifier_topic.py   \  
  --task_name=concept  \  
  --do_train=true \  
  --do_predict=true \  
  --do_eval=true  \  
  --data_dir=dataset   \  
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt   \  
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json   \  
  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt   \  
  --max_seq_length=256  \  
  --train_batch_size=8  \  
  --learning_rate=2e-5   \  
  --num_train_epochs=3.0  \  
  --output_dir=output_topic/
 ```

## results

the results which are labeled by the volunteers are released on results/

## reference

- transformers: <https://github.com/huggingface/transformers>
