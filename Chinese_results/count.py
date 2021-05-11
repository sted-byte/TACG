# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:41:14 2021

@author: yuansiyu
"""

count = 0
with open('t5pegasus_classification_manual.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        info, raw, cons = data.replace('\n','').split('------')
        con_ls = cons.split(',')
        if con_ls[0] == '':
            continue
        for con in con_ls:
            if con[-1] == '2' and con[:-1] not in info:
                print(con)
                count = count + 1