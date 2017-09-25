# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:38:17 2017
把该文件夹下的train_corpus_small文件中的所有文件都读取出来并分词
存于train_corpus_small_sep中
@author: zzd
"""
import os
import jieba
import pickle



#读取文件并返回类容
def readFile(path):
    with open(path,'rb') as f:
        contents=f.read()
        
    return contents

def writeFile(path,contents):
    with open(path,'wb') as f:
        f.write(contents)

#中文分词
def doSeg():       
    rootPath='train_corpus_small/'
    rootPathSeg='train_corpus_small_sep/'
    
    orgPaths=os.listdir(rootPath)
    for orgPath in orgPaths:
        print(orgPath)
        filePaths=os.listdir(rootPath+orgPath+'/')
        for filePath in filePaths:
            contents=readFile(rootPath+orgPath+'/'+filePath).strip()#去除多余的空格
            contents=contents.decode('utf-8').replace('\r\n','').strip()#去除windows下的换行符并去除多余空格
            contentSeg=jieba.cut(contents)#分词算法，用的是基于概率图的条件随机场（CRF）
            
            if not os.path.exists(rootPathSeg+orgPath+'/'):
                os.makedirs(rootPathSeg+orgPath+'/')
            
            writeFile(rootPathSeg+orgPath+'/'+filePath,' '.join(contentSeg).encode('utf-8'))
    print('中文语料分词结束！')   

from sklearn.datasets.base import Bunch

#Bunch类提供一种键值对的对象形式
#target_name:所有分类集名称列表
#label:每个文件的分类标签列表
#filenames:文件路径
#contents:分词后文件词向量形式
bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])
rootPathSeg='train_corpus_small_sep/'

orgPaths=os.listdir(rootPathSeg)
bunch.target_name.append(orgPaths)
for orgPath in orgPaths:
    filePaths=os.listdir(rootPathSeg+orgPath+'/')
    for filePath in filePaths:
        bunch.label.append(orgPath)
        fileFullPath=rootPathSeg+orgPath+'/'+filePath
        bunch.filenames.append(fileFullPath)
        bunch.contents.append(readFile(fileFullPath).strip())

#Bunch对象持久化
wordBagOrgPath="train_word_bag/"
wordBagPath=wordBagOrgPath+"train_set.dat"
if not os.path.exists(wordBagOrgPath):
    os.makedirs(wordBagOrgPath)
fileObj=open(wordBagPath,'wb',0)
pickle.dump(bunch,fileObj)
fileObj.close()

print("构建文本对象结束！")



