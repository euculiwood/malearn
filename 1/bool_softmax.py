import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import math
import random
import scipy.sparse as sp
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
time_start=time.time()
#{'电脑': 1, '法律': 2, '教育': 3, '经济': 4, '体育': 5, '政治': 6}
df_news1=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/1computer.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news1=df_news1.dropna()
df_news2=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/2law.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news2=df_news2.dropna()
df_news3=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/3education.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news3=df_news3.dropna()
df_news4=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/4economic.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news4=df_news4.dropna()
df_news5=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/5sport.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news5=df_news5.dropna()
df_news6=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/train/6politic.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
df_news6=df_news6.dropna()

ts_news1=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/1computer.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news1=ts_news1.dropna()
ts_news2=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/2law.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news2=ts_news2.dropna()
ts_news3=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/3education.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news3=ts_news3.dropna()
ts_news4=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/4economic.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news4=ts_news4.dropna()
ts_news5=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/5sport.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news5=ts_news5.dropna()
ts_news6=pan.read_table('/home/eucliwood/maler/NoteTest/Tsinghua/test/6politic.txt',na_values=['<text>','</text>'],names=['text'],encoding='utf-8')
ts_news6=ts_news6.dropna()

stopwords=pan.read_csv('/home/eucliwood/maler/NoteTest/Tsinghua/stop_words_zh.txt',encoding='utf-8',sep='\t',quoting=3,index_col=False,names=['stopword'])
stopwords.head()
stopwords=stopwords.stopword.values.tolist()


def split_words(content):
    for line_index in range(len(content)) :
        content[line_index] = content[line_index].split()
    return content


def drop_stopwords(contents,stopwords):
    contents_clean=[]
    all_words=[]
    a = 0
    for line in contents:
        line_clean=[]
        for word in line:
            if word in stopwords or word == ' ' or word == ',':
                continue
            line_clean.append(word)
            all_words.append(word)
        contents_clean.append(line_clean)
    return contents_clean,all_words


def create_vocablist(dataset):
    vocabset = set([])  # 创建一个空集
    for line in dataset:
            vocabset = vocabset | set(line)  # 创建两个集合的并集
    return list(vocabset)


def get_c_word_list(dfnews,need):
    after_split = split_words(dfnews.text.values.tolist())
    #new_content = pan.DataFrame({'contents':after_split})
    contents_clean, all_words = drop_stopwords(after_split, stopwords)
    if need ==1:
       return contents_clean
    if need ==2:
        return all_words


#词表
df_new = pan.concat([df_news1,df_news2])
df_new = pan.concat([df_new,df_news3])
df_new = pan.concat([df_new,df_news4])
df_new = pan.concat([df_new,df_news5])
df_new = pan.concat([df_new,df_news6])

content=df_new.text.values.tolist()
content = split_words(content)
df_content=pan.DataFrame({'content_S':content})
contents = df_content.content_S.values.tolist()
contents_clean,all_words=drop_stopwords(contents,stopwords)
words_list = create_vocablist(contents_clean)

time_end=time.time()
print('words list time',time_end-time_start)

label_list = []
v_list = []
label_test = []
v_test = []
'''
def turn_vector(df,label,train):
   words_c = get_c_word_list(df, 1)
   words_c_all = get_c_word_list(df_new, 1)
   time_end = time.time()
   print('vector get words cost', time_end - time_start)
   for i in range(len(df)):
       vec = [0] * len(words_list)
       for j in range(len(words_list)):
           if words_list[j] in words_c[i]:
               tf = words_c[i].count(words_list[j])/len(words_c[i])
               D = len(df_new)
               doc_num = 0
               for x in range(D):
                   if words_list[j] in words_c_all[x]:
                       doc_num += 1
               idf = math.log(D/(doc_num+1))
               tf_idf = tf * idf
               vec[j] = tf_idf
       if train == 1:
           v_list.append(vec)
           label_list.append(label)
       if train == 0:
           v_test.append(vec)
           label_test.append(label)

'''
def turn_vector(df,label,train,type):
    words_c = get_c_word_list(df, 1)
    words_c_all = get_c_word_list(df_new, 1)
    time_end = time.time()
    print('vector get words cost', time_end - time_start)
    if type == 'bool':
       for i in range(len(df)):
           vec = [0] * len(words_list)
           for j in range(len(words_list)):
               if words_list[j] in words_c[i]:
                   w = 1/10
                   vec[j] = w
           if train == 1:
               v_list.append(vec)
               label_list.append(label)
           if train == 0:
               v_test.append(vec)
               label_test.append(label)


def get_train_data(type):
    turn_vector(df_news1,1,1,type)
    turn_vector(df_news2,2,1,type)
    turn_vector(df_news3,3,1,type)
    turn_vector(df_news4,4,1,type)
    turn_vector(df_news5,5,1,type)
    turn_vector(df_news6,6,1,type)

get_train_data('bool')

#P(y=j|x;theta) = e^(theta_j^T x)/sum_1^C(e_j^T x)
def softmax(theta_x):
    sum_theta_x = np.sum(np.exp(theta_x),axis=1,keepdims=True)
    res = np.exp(theta_x)/sum_theta_x
    return res


def one_hot_arrary(label,samples_n,class_n):
    oha = np.zeros((samples_n,class_n))
    oha[np.arange(samples_n),label.T-1] = 1
    return  oha


alpha = 0.1
iters = 1000
classes = 6


def train(data,label,classes,iters,alpha):
    samples_n = data.shape[0]
    features_n = data.shape[1]
    #data = sp.csr_matrix(data)
    weights = np.random.rand(classes, features_n) #C*M
    loss_list = []
    y = one_hot_arrary(label,samples_n,classes)
    xt = [0]
    yg = [0]
    plt.ion()
    print(data)
    for i in range(300):
        #if(i > 3):
         #   if(loss_list[i-1]  < 0.2):3
          #         print("end")
           #        break
        theta_x = np.dot(data,weights.T)

        h_x = softmax(theta_x)

        #sigma_1^N sigma_1^C 1{y^k = j}log hj(x)
        loss =(-1.0/samples_n)* np.sum(y * np.log(h_x))
        print(loss)
        loss_list.append(loss)

        index = random.randint(0,samples_n-1)
        weights = weights + alpha * np.dot((y-h_x).T,data)

        time_end = time.time()
        print('flesh weights time', time_end - time_start)
        print("iters:",i+1)
        #plt.scatter(range(len(loss_list)),loss_list,200,marker='.',label='loss')
        if i % 5 ==0:
            xt[0] = i
            yg[0] = loss
            plt.scatter(xt,yg,c='b',)
            plt.pause(0.01)
    return weights,loss_list

def test(ts,weights):
    theta_x = np.dot(ts,weights.T)
    pos = softmax(theta_x)
    c_list = np.argmax(pos,axis=1).reshape((-1,1))
    return c_list

v_array = np.array(v_list)
label_array = np.array(label_list)

time_end=time.time()
print('train start',time_end-time_start)
weights,loss_list =train(v_array,label_array,classes,iters,alpha)

def get_test_data(type):
    turn_vector(ts_news1,0,0,type)
    turn_vector(ts_news2,1,0,type)
    turn_vector(ts_news3,2,0,type)
    turn_vector(ts_news4,3,0,type)
    turn_vector(ts_news5,4,0,type)
    turn_vector(ts_news6,5,0,type)


get_test_data('bool')
a_test = np.array(v_test)
label_test = np.array(label_test).reshape((-1,1))
y_predict = test(a_test,weights)

right = 0
for i in range(len(label_test)):
    if y_predict[i] == label_test[i]:
        right+=1
accuray = right/len(label_test)
print(accuray)
print(right)
print(len(label_test))

