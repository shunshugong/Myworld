'''训练文件 uid itemid rating 
   映射文件 item itemid'''
#from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.model_selection import GridSearchCV
from surprise import KNNBaseline
import pandas as pd
import os
#file_path = os.path.expanduser('~/data')       #不能识别中文，需要转成id号
#reader = Reader(line_format='user item rating', sep='\t')
#data = Dataset.load_from_file(file_path, reader=reader)
 #data.split(n_folds=3)
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()#不分割，全部训练
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)
#param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#              'reg_all': [0.4, 0.6]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
#gs.fit(trainset) #训练好的模型
def read_item_names():#id 与电影片名映射
    item_path=(r'C:\Users\顺叔公\.surprise_data\ml-100k\ml-100k\u.item')
    rid_to_name={}
    name_to_rid={}
    with open(item_path,'r',encoding='ISO-8859-1') as f:
        for line in f:
            line=line.split('|')
            rid_to_name[line[0]]=line[1]
            name_to_rid[line[1]]=line[0]
    return rid_to_name,name_to_rid
def high_recommend(uid):
    data_path=(r'C:\Users\顺叔公\.surprise_data\ml-100k\ml-100k/u.data')
    rec=[]
    with open(data_path,'r',encoding='ISO-8859-1') as f:
        for line in f:
            line=line.split('\t')
            if(int(line[0])==uid and int(line[2])>=3):
                rec.append(line[1])
    return rec#找到用户评分高的电影
rec=high_recommend(196)

rid_to_name, name_to_rid = read_item_names()
mov_list=[]
for mov in rec:
 mov_inner_id=algo.trainset.to_inner_iid(mov)
 mov_neighbor=algo.get_neighbors(mov_inner_id,k=2)
 mov_neighbor=(algo.trainset.to_raw_iid(inner_id) 
                       for inner_id in mov_neighbor)
 for rid in mov_neighbor:
  new_mov=rid_to_name[rid]
  mov_list.append(new_mov)
 if (len(mov_list)>=10):
  break