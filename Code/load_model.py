#coding: utf-8
import numpy as np
import joblib


flag="sklearn"

def text_features(text, feature_words):
    text_words = set(text)
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk特征 dict
        features = {word: 1 if word in text_words else 0 for word in feature_words}
    elif flag == 'sklearn':
        ## sklearn特征 list
        features = [1 if word in text_words else 0 for word in feature_words]
    else:
        features = []
    ## -----------------------------------------------------------------------------------
    return features


classifier=joblib.load("train_model.m")

title=input("请输入标题")
content=input("请输入正文")

#title="历史上此人昏庸无道，水浒里却忧国忧民，说岳中揭示真相"
#content="原标题：历史上此人昏庸无道，水浒里却忧国忧民，说岳中揭示真相无论是《水浒传》还是《说岳全传》都是根据两宋之交的一些历史事件改编、创作而出的历史小说，因此书中也不可避免地出现不少真实存在的历史人物和事件。之前我介绍了几位既在历史上真实存在、同时又在这两部小说中出现的人物，今天要说的是这个系列的最后一篇。这个人物便是宋徽宗赵佶。宋徽宗赵佶，在历史上既是一位多才多艺的书画巨匠，也是昏庸无道的帝王。在他统治北宋的二十五年间，文化、艺术等到了巨大的发展，同时也造成民不聊生、叛乱四起的混乱局面。后来金军入侵时，他与其子钦宗一起被俘，被押往五国城囚禁，北宋因此灭亡。他也堪称是中国历史上最为屈辱的皇帝之一。不过，宋徽宗在文艺作品中的形象与历史形象还是有一定距离的。在《水浒传》中，描述宋徽宗之处大概有三十多处，尽管着墨不多，但形象却较为饱满。虽说任用了蔡京、高俅、童贯等奸臣，又有与名妓李师师之间的丑事，但总体来看还是忧国忧民的。因此，书中经常出现他大骂高俅等人的话语，并有招安梁山这样“辉煌”的成就出现。《水浒传》中的宋徽宗，对于梁山叛乱的态度是坚决的，主张镇压。但得知宋江等人有意招安，便先后派出多位大臣前往梁山。梁山招安后，宋徽宗对梁山的态度总体上也是较为支持的。最终得知宋江、卢俊义被高俅等奸臣害死，还忍不住痛斥奸臣，并下诏为梁山众兄弟建造庙宇进行纪念，足见在内心深处对梁山好汉还是非常敬佩的。相对于《水浒传》中形象较为正面的宋徽宗而言，《说岳全传》中的他就被贬低了不少。书中说北宋之所以灭亡，是因为他在元旦祭天是错把“玉皇大帝”四个字写成了“王皇犬帝”，引得上天震怒，最终导致亡国。这个情节看似荒诞，其实是对其在历史上只知痴迷书画而无心治国的巨大讽刺。书中还提到，在宋徽宗的统治之下，朝政腐败，奸佞横行。面对金军大兵压境，宋徽宗惊慌失措，不敢奋起抵抗，反倒屈膝投降，最终被金军押往北方惨死。正如书中第十八回的诗中提到的：“徽钦二帝，老死沙漠之乡；义士忠臣，尽丧奸臣之手”，而这一切，都是宋徽宗自己亲手造成的。参考书籍：《水浒传》、《说岳全传》"
str_news=title+" "+content

all_words_npy=np.load("./data/all.npy")
print(all_words_npy)
all_words_list=all_words_npy.tolist()
#print(all_words_list)

test_feature=text_features(str_news,all_words_list)
#print(test_feature)

test_feature_list=[]
test_feature_list.append(test_feature)

print("概率为：")
print(classifier.predict_proba(test_feature_list))
strtype=["财经","房产","教育","军事","科技","汽车","体育","游戏","娱乐"]
print("分类为：")
print(strtype[int(classifier.predict(test_feature_list)[0])-1])


