import numpy as np
import pandas as pan
import matplotlib as mat
import jieba
import jieba.analyse
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

df_news1['label']='1'
df_news2['label']='2'
df_news3['label']='3'
df_news4['label']='4'
df_news5['label']='5'
df_news6['label']='6'

ts_news1['label']='1'
ts_news2['label']='2'
ts_news3['label']='3'
ts_news4['label']='4'
ts_news5['label']='5'
ts_news6['label']='6'
#print (df_news1)
#rint(len(df_news1))
#print(df_news.shape)
N1 = len(df_news1)
N2 = len(df_news2)
N3 = len(df_news3)
N4 = len(df_news4)
N5 = len(df_news5)
N6 = len(df_news6)
print(ts_news1)
print(ts_news2)

df_new = pan.concat([df_news1,df_news2])
df_new = pan.concat([df_new,df_news3])
df_new = pan.concat([df_new,df_news4])
df_new = pan.concat([df_new,df_news5])
df_new = pan.concat([df_new,df_news6])


ts_new = pan.concat([ts_news1,ts_news2])
ts_new = pan.concat([ts_new,ts_news3])
ts_new = pan.concat([ts_new,ts_news4])
ts_new = pan.concat([ts_new,ts_news5])
ts_new = pan.concat([ts_new,ts_news6])

content=df_new.text.values.tolist() # 将每条新闻内容组成列表
#content_test = ts_new.text.values.tolist()

def split_words(content):
    for line_index in range(len(content)) :
        content[line_index] = content[line_index].split()
    return content

content = split_words(content)
#print(content[0])


df_content=pan.DataFrame({'content_S':content})
df_content.head()

#ts_content=pan.DataFrame({'content_S':content_tS})
#ts_content.head()
#print(df_content)
#print(ts_content)


#读取停止符
stopwords=pan.read_csv('/home/eucliwood/maler/NoteTest/Tsinghua/stop_words_zh.txt',encoding='utf-8',sep='\t',quoting=3,index_col=False,names=['stopword'])
stopwords.head()


##去除停止词
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


#构造剔除之后的词表和停止词表
contents = df_content.content_S.values.tolist()

stopwords=stopwords.stopword.values.tolist()
#运行剔除函数
contents_clean,all_words=drop_stopwords(contents,stopwords)
df_content=pan.DataFrame({'contents_clean':contents_clean})
df_content.head()
#print(all_words)
#ts_content=pan.DataFrame({'contents_clean':ts_contents_clean})
#ts_content.head()
#print(ts_content)

'''
def take_class_word(df):
    a=[]
    df_list = df.contents_clean.values.tolist()
    for line in df_list:
        for word in line:
            a.append(word)
    return  a
'''

#得到训练数据
df_train=pan.DataFrame({'contents_clean':contents_clean,'label':df_new['label']})
df_train.head()
#print(df_train)

#测试集数据
#df_test=pan.DataFrame({'contents_clean':ts_contents_clean,'label':ts_new['label']})
#df_test.head()
#print(df_test)


#改变文本形式，用空格连接每个词
def take(contents):
    words = []
    for line in contents:
        words.append(' '.join(line))
    return words


x_train=df_train.contents_clean.values.tolist()
y_train = df_train.label.values.tolist()
#print(n_contents)
words = take(x_train)
#print(len(words))

#x_test=df_test.contents_clean.values.tolist()
#y_test=df_test.label.values.tolist()
#words_test=take(x_test)

#words
#y_train
#words_test
#y_test


# 创建一个包含在所有文档中出现不重复的词的列表，这个词汇表可以根据现实情况给出，可以人为选择设置，性质和特征类似
def create_vocablist(dataset):
    vocabset = set([])  # 创建一个空集
    for line in dataset:
            vocabset = vocabset | set(line)  # 创建两个集合的并集
    return list(vocabset)


#词表
words_list = create_vocablist(contents_clean)
word_p=pan.DataFrame({'contents_clean':words_list})
#print(word_p)
#print(words_list)

def get_c_word_list(dfnews,need):
    after_split = split_words(dfnews.text.values.tolist())
    #new_content = pan.DataFrame({'contents':after_split})
    contents_clean, all_words = drop_stopwords(after_split, stopwords)
    if need ==1:
       return contents_clean
    if need ==2:
        return all_words


words_c1 = get_c_word_list(df_news1,2)
words_c2 = get_c_word_list(df_news2,2)
words_c3 = get_c_word_list(df_news3,2)
words_c4 = get_c_word_list(df_news4,2)
words_c5 = get_c_word_list(df_news5,2)
words_c6 = get_c_word_list(df_news6,2)


def check_frq(content, word_all):
    pc_list = []
    for w in content:
        #计算频度
        pc_list.append((word_all.count(w)+ 1)/(len(word_all)+len(content)))
    return pc_list


pc1_ls=check_frq(words_list,words_c1)
pc2_ls=check_frq(words_list,words_c2)
pc3_ls=check_frq(words_list,words_c3)
pc4_ls=check_frq(words_list,words_c4)
pc5_ls=check_frq(words_list,words_c5)
pc6_ls=check_frq(words_list,words_c6)
#word_p=pan.DataFrame({'contents_clean':words_list,'c1':pc1_ls})
#right = 0
#false = 0


P_C1=len(words_c1)/len(all_words)
P_C2=len(words_c2)/len(all_words)
P_C3=len(words_c3)/len(all_words)
P_C4=len(words_c4)/len(all_words)
P_C5=len(words_c5)/len(all_words)
P_C6=len(words_c6)/len(all_words)


a=[]
a.append(pc1_ls)
a.append(pc2_ls)
a.append(pc3_ls)
a.append(pc4_ls)
a.append(pc5_ls)
a.append(pc6_ls)
pos = np.array(a)
PC = np.array([P_C1,P_C2,P_C3,P_C4,P_C5,P_C6])
pos = np.dot(pos,10000)

#print(len(words_list))


def test_fun(df,label):
    right = 0
    false = 0
    cl=[]
    news_list = get_c_word_list(df,1)
    for i in range(len(df)):
        a = np.array([1, 1, 1, 1, 1, 1])
        a=a.astype(float)
        for j in range(0,6):
            a[j] *= PC[j]

        for word in news_list[i]:
            try:
                index = words_list.index(word)
            except:
                continue
            for j in range(0,6):
                   a[j] *= pos[j][index]
        c = np.argmax(a)+1
        cl.append(c)
        if c == label:
            right += 1
        else:
            false += 1
    print(right)
    print(false)
    print(cl)
    return right/len(df)

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