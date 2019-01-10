# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:51:48 2018

@author: hzp0625
"""

import pandas as pd
import os
os.chdir('F:\\python_study\\pachong\\斗破苍穹')
import re
import numpy as np
import jieba
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc')#,size=20指定本机的汉字字体位置
import matplotlib.pyplot as plt
import networkx as nx  


texts = open('all（校对版全本）.txt',"r")

texts = texts.read()

AllChapters = re.split('第[0-9]*章',texts)[1:]

AllChapters = pd.DataFrame(AllChapters,columns = ['text'])
AllChapters['n'] = np.arange(1,1647)

# 载入搜狗细胞词库
jieba.load_userdict('斗破苍穹.txt')
jieba.load_userdict('斗破苍穹异火.txt')
stopwords = open('中文停用词表（比较全面，有1208个停用词）.txt','r').read()
stopwords = stopwords.split('\n')




# 分章节分词
def fenci(x):
    cut_text = " ".join([w for w in jieba.cut(x) if w not in stopwords])
    return cut_text


# 先取100个尝试
#AllChapters = AllChapters.loc[:120]
result = pd.DataFrame(AllChapters.text.apply(fenci))    
result.columns = ['fenci']
   
          
# 女主每章出现次数统计：熏儿，云韵，小医仙，彩鳞，美杜莎

names = ['熏儿','云韵','小医仙','彩鳞','美杜莎']
result['熏儿'] = result.fenci.apply(lambda x:x.count('熏儿') + x.count('薰儿'))
result['云韵'] = result.fenci.apply(lambda x:x.count('云韵'))
result['小医仙'] = result.fenci.apply(lambda x:x.count('小医仙'))
result['彩鳞'] = result.fenci.apply(lambda x:x.count('彩鳞') + x.count('美杜莎'))


plt.figure(figsize=(15,5))
plt.plot(np.arange(1,result.shape[0]+1),result['熏儿'],color="r",label = u'熏儿')
plt.plot(np.arange(1,result.shape[0]+1),result['云韵'],color="lime",label = u'云韵')
plt.plot(np.arange(1,result.shape[0]+1),result['小医仙'],color="gray",label = u'小医仙')
plt.plot(np.arange(1,result.shape[0]+1),result['彩鳞'],color="orange",label = u'彩鳞')
plt.legend(prop =font)
plt.xlabel(u'章节',fontproperties = font)
plt.ylabel(u'出现次数',fontproperties = font)
plt.show()

   




# 主要人物出现的总频数，人物名单从百度百科获取
nameall = open('所有人物.txt','r').read().split('\n')
nameall = pd.DataFrame(nameall,columns = ['name'])
textsall = ''.join(AllChapters.text.tolist())
nameall['num'] = nameall.name.apply(lambda x:textsall.count(x))

nameall.loc[nameall.name=='熏儿','num'] = nameall.loc[nameall.name=='熏儿','num'].values[0] + nameall.loc[nameall.name=='熏儿','num'].values[0]
nameall.loc[nameall.name=='熏儿','num'] = -886



nameall.loc[nameall.name=='彩鳞','num'] = nameall.loc[nameall.name=='彩鳞','num'].values[0] + nameall.loc[nameall.name=='美杜莎','num'].values[0]
nameall.loc[nameall.name=='美杜莎','num'] = -886

nameall = nameall.sort_values('num',ascending = False)


plt.figure(figsize=(8,10))
fig = plt.axes()
n = 50
plt.barh(range(len(nameall.num[:n][::-1])),nameall.num[:n][::-1],color = 'darkred')
fig.set_yticks(np.arange(len(nameall.name[:n][::-1])))
fig.set_yticklabels(nameall.name[:n][::-1],fontproperties=font)
plt.xlabel('人物出场次数',fontproperties = font)
plt.show()


    
# 社交网络图  共现矩阵
# 两个人物出现在同一段，说明有某种关系
words = open('all（校对版全本）.txt','r').readlines()
words = pd.DataFrame(words,columns = ['text'],index = range(len(words)))
words['wordnum'] = words.text.apply(lambda x:len(x.strip()))
words = words.loc[words.wordnum>20,]
wrods = words.reset_index(drop = True)
relationmat = pd.DataFrame(index = nameall.name.tolist(),columns = nameall.name.tolist()).fillna(0)


wordss = words.text.tolist()
for k in range(len(wordss)):
    for i in nameall.name.tolist():
        for j in nameall.name.tolist():
            if i in wordss[k] and j in  wordss[k]:
                relationmat.loc[i,j] += 1 
    if k%1000 ==0:
        print(k)
    
relationmat.to_excel('共现矩阵.xlsx')

# 网络图


# 边与权重矩阵
#relationmat1 = pd.DataFrame(index = range(relation.shape[]))
relationmat1 = {}
for i in relationmat.columns.tolist():
    for j in relationmat.columns.tolist():
        relationmat1[i, j] = relationmat.loc[i,j]


edgemat = pd.DataFrame(index = range(len(relationmat1)))
node = pd.DataFrame(index = range(len(relationmat1)))

edgemat['Source'] = 0
edgemat['Target'] = 0
edgemat['Weight'] = 0

node['Id'] = 0
node['Label'] = 0
node['Weight'] = 0


names = list(relationmat1.keys())
weights = list(relationmat1.values())
for i in range(edgemat.shape[0]):
    name1 = names[i][0]
    name2 = names[i][1]
    if name1!=name2:
        edgemat.loc[i,'Source'] = name1
        edgemat.loc[i,'Target'] = name2
        edgemat.loc[i,'Weight'] = weights[i]
    else:
        node.loc[i,'Id'] = name1
        node.loc[i,'Label'] = name2
        node.loc[i,'Weight'] = weights[i]        
    i+=1


edgemat = edgemat.loc[edgemat.Weight!=0,]
edgemat = edgemat.reset_index(drop = True)
node = node.loc[node.Weight!=0,]
node = node.reset_index(drop = True)



edgemat.to_csv('边.csv',index = False)
node.to_csv('节点.csv',index = False)








