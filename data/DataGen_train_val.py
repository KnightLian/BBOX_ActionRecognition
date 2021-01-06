import os
import torch
import sys
import pandas as pd
import numpy as np
import torch.nn as n

###################################################################################
# create training and validating data
def givelist(abovefolder):
    lst = os.listdir(abovefolder)
    lst = sorted(lst)
    vidlist = [os.path.join(abovefolder,ii) for ii in lst]
    return vidlist

def showframe(frame):
    f = open(frame,'r')
    mat = []
    for x in f:
        eachline = x.split()
        conveachline = [float(ii) for ii in eachline]
        mat.append(conveachline)
    f.close
    return mat

def makeframetemplate(framedatalist):
    labelstatus = ['shou','shiguan','shiguanjiazi','bolibang','shaobei']
    labelstatus = {eachlabel:i for i,eachlabel in enumerate(labelstatus)}

    # create frame template
    frrow = ['shou','shou', 'shiguan','shiguanjiazi','bolibang','shaobei'] #rows we want
    frrow = [labelstatus[i] for i in frrow]
    frcol = ['label', 'score', 'cx', 'cy', 'w', 'h']
    frtable = pd.DataFrame([-1]*len(frcol) for _ in range(len(frrow))).astype("float") #create all -1
    frtable.columns = frcol
    frtable['label'] = frrow
    frtable['label'] = frtable['label'].astype("int")
#     display(frtable)

    # add raw data from frame
    frtable2 = pd.DataFrame(framedatalist)
    frtable2.columns = frcol
    frtable2 = frtable2[frtable2['score']>0.7] #score 0.7以下的都删了
    frtable2['label'] = frtable2['label'].astype("int")
    frtable2 = frtable2.sort_values(by='label', ascending=True)
#     display(frtable2)

    frtable = frtable.values.tolist()
    frtable2 = frtable2.values.tolist()
    for i in range(len(frtable)):
        for j in range(len(frtable2)):
            if frtable[i][0] == frtable2[j][0]:
                frtable[i] = frtable2.pop(j)
                break

    frtable = pd.DataFrame(frtable)
    frtable.columns = frcol
    frtable['label'] = frrow
    frtable['label'] = frtable['label'].astype("int")
    frtable = pd.concat([pd.get_dummies(frtable['label']), frtable], axis=1)
    frtable = frtable.drop(columns='label')
    frtable = frtable.drop(columns='score')
#     display(frtable)

    return frtable.values.tolist()

def makefmtensor(framedirectory, stircodeforframe):
    frames = givelist(framedirectory)
    temp = []
#     framecodelist = []
    for ff in frames:
        rawf2data = showframe(ff)
#         print(rawf2data)
        rawf2data = makeframetemplate(rawf2data)
#         print(rawf2data)
        temp.append(rawf2data)
# #         framecodelist.append(stircodeforframe) ##1个视频的每一帧接上一个label

    framecodelist = stircodeforframe  #1个视频 接上一个label

    eachtensor = torch.tensor(temp)

#     print(eachtensor[0])
#     print(eachtensor.size())
    eachtensor = torch.flatten(eachtensor, 1, 2) # [76, 6, 9] to [76, 54]
#     print(eachtensor[0])
#     print(eachtensor.size())

    eachtensor = eachtensor[None, None, :, :] #torch.Size([1, 1, 76, 54])
#     print(eachtensor.size())
    eachtensor = eachtensor.float()

    videoframes = torch.tensor(framecodelist)
#     print(videoframes.size(), videoframes)
    videoframes = videoframes.float()

    return eachtensor, videoframes

def txt2tensor(psive, stir, camera, stircode): # video list of example positve and stir
    videos = os.path.join(psive, stir)
    videos = givelist(videos)
    frontv = [os.path.join(vv, camera) for vv in videos]
    videos = []
    for i in range(len(frontv)):
        event, label = makefmtensor(frontv[i], stircode)
        aa = (event, label)
        videos.append(aa)
    return videos

# Initialize Data Received from ShangHai
defclass = ['溶解_搅拌','溶解度实验-固体缓慢竖起','溶解_上试管夹', '错的动作'] #定义对的动作，错了各种动作合并为一个错的动作
defclass = set(defclass)
defclass = sorted(defclass)
defclass = {word:i+1 for i,word in enumerate(defclass)}
### print(defclass) # {'溶解_上试管夹': 1, '溶解_搅拌': 2, '溶解度实验-固体缓慢竖起': 3, '错的动作': 4}

##########positive_data##########################
maindir = './动作识别-front-txt/动作正样本' # 打开对的动作的文件夹创建数据集
camera = 'front'

action = '溶解_搅拌'
actcode = defclass[action]
posdata1 = txt2tensor(maindir, action, camera, actcode)

action = '溶解度实验-固体缓慢竖起'
actcode = defclass[action]
posdata2 = txt2tensor(maindir, action, camera, actcode)

action = '溶解_上试管夹'
actcode = defclass[action]
posdata3 = txt2tensor(maindir, action, camera, actcode)
# ##########negative_data##########################
maindir = './动作识别-front-txt/动作负样本'
camera = 'front'

action = '错的动作'
actcode = defclass[action]

action = '溶解_搅拌'
negdata1 = txt2tensor(maindir, action, camera, actcode)
action = '溶解度实验-固体缓慢竖起'
negdata2 = txt2tensor(maindir, action, camera, actcode)
action = '溶解_上试管夹'
negdata3 = txt2tensor(maindir, action, camera, actcode)

# ########## shuffle initial data ##################
for _ in range(50):
    np.random.shuffle(posdata1)
    np.random.shuffle(posdata2)
    np.random.shuffle(posdata3)

    np.random.shuffle(negdata1)
    np.random.shuffle(negdata2)
    np.random.shuffle(negdata3)

# ########## Split into train and validation ##################
def splittrainval(indata, rate):
    return indata[:int(rate*len(indata))], indata[int(rate*len(indata)):]

p1train, p1val = splittrainval(posdata1, 0.8)
p2train, p2val = splittrainval(posdata2, 0.8)
p3train, p3val = splittrainval(posdata3, 0.8)

n1train, n1val = splittrainval(negdata1, 0.8)
n2train, n2val = splittrainval(negdata2, 0.8)
n3train, n3val = splittrainval(negdata3, 0.8)
###########merge_videos#############################
def pk1from(n1,n2): #include n2
    pkednum = np.random.choice(range(n1, n2+1), size=1)
    return pkednum[0]

def pkvideo(p1indata, p2indata, p3indata, n1indata, n2indata, n3indata):
    posidict = {1:p1indata, 2:p2indata, 3:p3indata}
    negadict = {1:n1indata, 2:n2indata, 3:n3indata}
    pickpn = pk1from(0,1) #pick 0 or 1
    if pickpn == 1: #positive
        pickact = pk1from(1,len(posidict))
        pkvideol = posidict[pickact]
    else: # ==0 #negative
        pickact = pk1from(1,len(negadict))
        pkvideol = negadict[pickact]
#     print(len(pkvideol) )
    pickvnum = pk1from(0, len(pkvideol)-1) #starting from 0
#     print('pick p/n:',pickpn,', pick action:',pickact,', pick which video:',pickvnum)
    return pkvideol[pickvnum]

def mergev(video1, addvnum, p1indata, p2indata, p3indata, n1indata, n2indata, n3indata):
    global video2
    basev = video1[0]
    basel = [int(video1[1])]
    if addvnum != 0:
        for i in range(addvnum):
            video2 = pkvideo(p1indata, p2indata, p3indata, n1indata, n2indata, n3indata)
#             print('Add Video Size:', video2[0].size())
            basev = torch.cat((basev,video2[0]),2)
            basel.append(int(video2[1]))
    basel = torch.tensor(basel).float()
    return (basev,basel)

# Create training dataset from combining videos
indataraw = []
setsize = 4000 #生成2000个数据集，可以改成你想要的多少
for _ in range(setsize): #define data amount in dataset
    video1 = pkvideo(p1train, p2train, p3train, n1train, n2train, n3train)
#     print('Base Video Size:', video1[0].size())
    addvnum = pk1from(1,3) # target 4 actions in total  #在拼接多少个视频、根据测试集最多拼了4个动作所以这里不能拼接他少了，因为一个视频只有一个动作。
    newvideo = mergev(video1, addvnum, p1train, p2train, p3train, n1train, n2train, n3train)
    indataraw.append(newvideo)
#     print('combined video size:', newvideo[0].size(), newvideo[1].size())
#     print('===============================================')

# Create validation dataset from combininig rest videos
valdataraw = []
setsize = 400 #生成2000个数据集，可以改成你想要的多少
for _ in range(setsize): #define data amount in dataset
    video1 = pkvideo(p1val, p2val, p3val, n1val, n2val, n3val)
#     print('Base Video Size:', video1[0].size())
    addvnum = pk1from(1,3) # target 4 actions in total  #在拼接多少个视频、根据测试集最多拼了4个动作所以这里不能拼接他少了，因为一个视频只有一个动作。
    newvideo = mergev(video1, addvnum, p1val, p2val, p3val, n1val, n2val, n3val)
    valdataraw.append(newvideo)
#     print('combined video size:', newvideo[0].size(), newvideo[1].size())
#     print('===============================================')

indata = indataraw
valdata = valdataraw

# ###########Shuffle#################################
np.random.shuffle(indata)
np.random.shuffle(valdata)

torch.save(indata, "ARNDatasetTrainSingleV2.pt")
torch.save(valdata, "ARNDatasetValSingleV2.pt")

print('class: ', len(defclass)+1, 'training data size: ', len(indata), 'validating data size: ', len(valdata))

print('finish load traning and validating data')
