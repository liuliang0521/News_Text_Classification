import numpy as np

strtype=["财经","房产","教育","军事","科技","汽车","体育","游戏","娱乐"]


for l in range(0,9):
    data=np.load("./data/train_data/"+strtype[l]+".npy")
    size=len(data)
    print(data)
    print(size)
    counter=1
    for j in range(0,size):
        file = open("./Database/SogouC/Sample/" + str(l+1)+ "/"+str(counter)+".txt", "a")
        file.write(data[j])
        counter=counter+1
        file.close()