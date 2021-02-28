import numpy as np
import pandas as pan
import math
import matplotlib as mat
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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


N=len(df_news1) + len(df_news2) + len(df_news3) + len(df_news4) + len(df_news5)+len(df_news6)
P_C1=len(df_news1)/N
P_C2=len(df_news2)/N
P_C3=len(df_news3)/N
P_C4=len(df_news4)/N
P_C5=len(df_news5)/N
P_C6=len(df_news6)/N

def train(df):
    words_c = get_c_word_list(df, 1)
    pc_ls=[]
    for word in words_list:
        doc_num = 0
        for line in words_c:
            if word in line:
                doc_num +=1
        pc_ls.append((doc_num+1)/(len(df)+2))
    return pc_ls

pc1_ls = train(df_news1)
pc2_ls = train(df_news2)
pc3_ls = train(df_news3)
pc4_ls = train(df_news4)
pc5_ls = train(df_news5)
pc6_ls = train(df_news6)

a=[]
a.append(pc1_ls)
a.append(pc2_ls)
a.append(pc3_ls)
a.append(pc4_ls)
a.append(pc5_ls)
a.append(pc6_ls)
pos = np.array(a)
pos = np.dot(pos,1000)


#print(pos)
PC = np.array([P_C1,P_C2,P_C3,P_C4,P_C5,P_C6])
#print(PC)
#print(pos)

def test_fun(df,label):
    words_c = get_c_word_list(df, 1)
    right = 0
    false = 0
    cl=[]
    con = 0
    for i in range(len(df)):
        a = np.array([1, 1, 1, 1, 1, 1])
        a = a.astype(float)
        for j in range(0,6):
            a[j] += math.log(PC[j])
        for index in range(len(words_list)):
            if words_list[index] in words_c[i]:
                for x in range(0,6):
                    a[x] += math.log(pos[x][index])
            else:
                for x in range(0,6):
                    a[x] += math.log((1000-pos[x][index]))
            con+=1

           # if i == 3:
            #    print(a)
            if a[np.argmin(a)] < 1e-100:
                 var = 1
                 while var == 1:
                    b = np.dot(a,10)
                    if b[np.argmax(b)] > 1:
                        break
                    else:
                        a = b
            if a[np.argmax(a)] > 1e100:
                 var = 1
                 while var == 1:
                     b=np.dot(a,1e-1)
                     if b[np.argmin(b)] <1:
                        break
                     else:
                        a=b
        c = np.argmax(a) + 1
        cl.append(c)
        if c == label:
            right += 1
        else:
            false += 1
    print(right)
    print(false)
    print(cl)
    return right / len(df)

acc1 = test_fun(ts_news1,1)
print(acc1)

acc2 = test_fun(ts_news2,2)
print(acc2)
acc3 = test_fun(ts_news3,3)
print(acc3)
acc4 = test_fun(ts_news4,4)
print(acc4)
acc5 = test_fun(ts_news5,5)
print(acc5)
acc6 = test_fun(ts_news6,6)
print(acc6)
print('词表大小')
print(len(words_list))

print('综合准确率')
acuy = np.array([acc1,acc2,acc3,acc4,acc5,acc6])
print(np.mean(acuy))







