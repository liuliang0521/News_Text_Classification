#coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
import pylab as pl
import matplotlib.pyplot as plt

def TextProcessing(folder_path, test_size=0.2):#数据的预处理部分
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 1000: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file), 'r') as fp:
               raw = fp.read()
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    ## 划分训练集和测试集
    data_class_zip = zip(data_list, class_list)
    data_class_list=list(data_class_zip)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_words_npy=np.load("./data/all.npy")
    all_words_list=all_words_npy.tolist()
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list



def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)

        joblib.dump(classifier,"train_model.m")

        print(classifier.predict(test_feature_list))
        print(classifier.predict_proba(test_feature_list))

        #for test_feature in test_feature_list:
        #    print(classifier.predict(test_feature)[0].values.reshape(-1,1))
        # print ''
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy


if __name__ == '__main__':

    print("start")

    ## 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    # 导入无效词汇表，去掉重复
    # 生产去重词汇表

    '''
    all_words_npy=np.array(all_words_list,dtype=str)
    train_data_npy = np.array(train_data_list,dtype=str)
    test_data_npy = np.array(test_data_list,dtype=str)
    train_class_npy = np.array(train_class_list,dtype=str)
    test_class_npy = np.array(test_class_list,dtype=str)

    np.save("./data/all_words.npy",all_words_npy)
    np.save("./data/train_data.npy", train_data_npy)
    np.save("./data/test_data.npy", test_data_npy)
    np.save("./data/train_class.npy", train_class_npy)
    np.save("./data/test_class.npy", test_class_npy)

    all_words_list=np.load("./data/all_words.npy").tolist()
    train_data_list = np.load("./data/train_data.npy").tolist()
    test_data_list = np.load("./data/test_data.npy").tolist()
    train_class_list = np.load("./data/train_class.npy").tolist()
    test_class_npy = np.load("./data/test_class.npy").tolist()
    '''
    #print(all_words_list)
    #print(train_data_list)
    #print(test_data_list)
    #print(train_class_list)
    #print(test_class_list)


    ## 文本特征提取和分类
    #flag = 'nltk'
    flag = 'sklearn'
    test_accuracy_list = []
    feature_words=all_words_list
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)


    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    print(test_accuracy)

    print("finished")