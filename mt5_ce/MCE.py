import torch.nn as nn
import numpy as np
import torch

class MCE(nn.Module):
    def __init__(self,
                 model,
                 model_dim,
                 classification_dim
                 ):
        super(MCE, self).__init__()
        self.model = model
        self.linear = nn.Linear(model_dim, classification_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.tanh=nn.Tanh()

    # 传入 input_ids=[batch_size,sen_len]
    #    labels=[batch_size,]
    #    labels_classification=[batch_classification]
    def forward(self, input_ids, attention_mask, labels, labels_classification):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        encoder_hidden_states = outputs["encoder_last_hidden_state"]
        labels_classification=labels_classification==1
        # encoder_hidden_states = encoder_hidden_states[labels_classification, 0] # 取第一个token
        encoder_hidden_states = torch.mean(encoder_hidden_states[labels_classification],1)
        x = self.linear(encoder_hidden_states)
        # x = self.sigmoid(x)
        x= self.tanh(x)
        # x = self.softmax(x)
        return outputs.loss, x

    def forward_t5pegasus(self,input_ids,attention_mask,labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def forward_test(self,input_ids,num_beams,is_mutual,sep_id):
        # temp=self.model.generate(input_ids=input_ids,output_attentions =True)
        temp = self.model.generate(input_ids,num_beams=num_beams,num_return_sequences=num_beams,early_stopping=True,output_scores=True,return_dict_in_generate=True)
        if num_beams!=1:
            result=temp[0]
        else:
            result=temp
        return temp

    def forward_mutual(self,input_ids,none_label):
        outputs=self.model(input_ids=input_ids,labels=none_label)
        encoder_hidden_states = outputs["encoder_last_hidden_state"]
        encoder_hidden_states= torch.mean(encoder_hidden_states[:],1)
        x = self.linear(encoder_hidden_states)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        _,indexs=torch.max(x,-1)
        return x[0][0].                                                                                                                                                                                                                                                                                                                                                                      item()
        return indexs.item()

    def getCrossDistribution(self,input_ids,labels,layer,head):
        result=self.model(input_ids=input_ids,labels=labels,output_attentions=True)
        distribution=result['cross_attentions'][layer-1][0][head-1]
        return distribution
