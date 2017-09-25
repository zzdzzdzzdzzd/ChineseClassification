# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 21:38:17 2017
把该文件夹下的train_corpus_small文件中的所有文件都读取出来并分词
存于train_corpus_small_sep中
@author: zzd
"""
import os
import jieba
import pickle#引入持久化类
from sklearn.datasets.base import Bunch#引入Bunch类
from sklearn.feature_extraction.text import TfidfTransformer#TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer#TF-IDF向量生成类
import random
from sklearn import metrics

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

#Bunch对象持久化
#写入Bunch对象
def writeBunchObj(path,bunchObj):
    fileObj=open(path,'wb')
    pickle.dump(bunchObj,fileObj)
    fileObj.close()

#读入Bunch对象
def readBunchObj(path):
    fileObj=open(path,'rb')
    bunch=pickle.load(fileObj)
    fileObj.close()
    return bunch

#构建文本对象
def createBunch():
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
    writeBunchObj(wordBagPath,bunch)
    
    print("构建文本对象结束！")

#创建训练集词空间
def createTrain():
    wordBagOrgPath="train_word_bag/"
    wordBagPath=wordBagOrgPath+"train_set.dat"
    #导入分词后的词向量
    bunch=readBunchObj(wordBagPath)
    #构建TF-IDF词向量空间对象
    tfidfSpace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    
    stopWordsPath='train_word_bag/hlt_stop_words.txt'
    stopWords=readFile(stopWordsPath)
    
    #初始化向量空间模型
    vectorizer=TfidfVectorizer(stop_words=stopWords,sublinear_tf=True,max_df=0.5)
    #文本转为词频矩阵，单独保存字典文件
    tfidfSpace.tdm=vectorizer.fit_transform(bunch.contents)
    tfidfSpace.vocabulary=vectorizer.vocabulary_
    #持久化向量词袋
    spacePath='train_word_bag/tfidf_space.dat'
    writeBunchObj(spacePath,tfidfSpace)

#随机在每个类别中抽取10个作为测试集
def randomTest():
    testRootOrgPath='test_corpus_small_sep/'
    trainRootOrgPath='train_corpus_small_sep/'
    if not os.path.exists(testRootOrgPath):
        os.makedirs(testRootOrgPath)
    
    trainPaths=os.listdir(trainRootOrgPath)
    for trainPath in trainPaths:
        if not os.path.exists(testRootOrgPath+trainPath):
            os.makedirs(testRootOrgPath+trainPath)
        
        trainFiles=os.listdir(trainRootOrgPath+trainPath)
        randomIndexs=random.sample(range(len(trainFiles)),10)#[random.randint(0,len(trainFiles)) for _ in range(10)]#random.randint(a,b)方法查收的随机数既包含a又包含b
        
        for randomIndex in randomIndexs:
            content=readFile(trainRootOrgPath+trainPath+'/'+trainFiles[randomIndex])
            writeFile(testRootOrgPath+trainPath+'/'+trainFiles[randomIndex],content)


def createTestBunch():
    #Bunch类提供一种键值对的对象形式
    #target_name:所有分类集名称列表
    #label:每个文件的分类标签列表
    #filenames:文件路径
    #contents:分词后文件词向量形式
    bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])
    rootPathSeg='test_corpus_small_sep/'
    
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
    wordBagOrgPath="test_word_bag/"
    wordBagPath=wordBagOrgPath+"test_set.dat"
    if not os.path.exists(wordBagOrgPath):
        os.makedirs(wordBagOrgPath)
    writeBunchObj(wordBagPath,bunch)
    
    print("构建文本对象结束！")

def createTest():
    wordBagOrgPath="test_word_bag/"
    wordBagPath=wordBagOrgPath+"test_set.dat"
    #导入分词后的词向量
    bunch=readBunchObj(wordBagPath)
    
    #导入训练集词袋
    trainBunch=readBunchObj('train_word_bag/tfidf_space.dat')
    #构建TF-IDF词向量空间对象
    tfidfSpace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    
    stopWordsPath='train_word_bag/hlt_stop_words.txt'
    stopWords=readFile(stopWordsPath)
    
    #初始化向量空间模型
    vectorizer=TfidfVectorizer(stop_words=stopWords,sublinear_tf=True,max_df=0.5,vocabulary=trainBunch.vocabulary)
    #文本转为词频矩阵，单独保存字典文件
    tfidfSpace.tdm=vectorizer.fit_transform(bunch.contents)
    tfidfSpace.vocabulary=trainBunch.vocabulary
    #持久化向量词袋
    spacePath='test_word_bag/test_space.dat'
    writeBunchObj(spacePath,tfidfSpace)
    
    
#导入多项式贝叶斯算法包
from sklearn.naive_bayes import MultinomialNB

#导入训练集向量空间
trainPath='train_word_bag/tfidf_space.dat'  
trainSet=readBunchObj(trainPath)

testPath='test_word_bag/test_space.dat'    
testSet=readBunchObj(testPath)    
    
model=MultinomialNB(alpha=0.001)   
model.fit(trainSet.tdm,trainSet.label) 
    
predicted=model.predict(testSet.tdm)

total=len(predicted)
rate=0

for flabel,fileName,expectCate in zip(testSet.label,testSet.filenames,predicted):
    if flabel!=expectCate:
        rate=rate+1
        print(fileName+"\t实际类别："+flabel+"\t预测类别："+expectCate)

print('错误率：'+str(rate/total))

def metrics_result(actual,predict):   
    print('准确度：{0:.3f}'.format(metrics.precision_score(actual,predict)))
    print('召回率：{0:.3f}'.format(metrics.recall_score(actual,predict)))
    print('F1值：{0:.3f}'.format(metrics.f1_score(actual,predict)))
    
    
#metrics_result(testSet.label,predicted)       
    
    
    
    
    
