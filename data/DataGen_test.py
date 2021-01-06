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
# Initialize Data Received from ShangHai

defclass = ['溶解_搅拌','溶解度实验-固体缓慢竖起','溶解_上试管夹', '错的动作'] #定义对的动作，错了各种动作合并为一个错的动作
defclass = set(defclass)
defclass = sorted(defclass)
defclass = {word:i+1 for i,word in enumerate(defclass)}
# print(defclass)

# ###########add test data#################################
# create testing data
def testtxt2tensor(psive, camera, acttype): # video list of example positve and stir
    videos = psive
    videos = givelist(videos)
    frontv = [os.path.join(vv, camera) for vv in videos]
    videos = []
    for i in range(len(frontv)):
        event, label = makefmtensor(frontv[i],acttype)
        aa = (event, label)
        videos.append(aa)
    return videos

def maketestdata(eachfolder, defclass):
    tempactdict = {1:'溶解度实验-固体缓慢竖起',2:'溶解_上试管夹',3:'溶解_搅拌'}
    fpath = os.path.join(maindir,eachfolder)
#     print(fpath)
    fcode = [int(oo) for oo in eachfolder.replace('-', ' ').split(' ')]
#     print(fcode)
    actlen = len(fcode)
    fcode = [tempactdict[oo] for oo in fcode]
#     print(fcode)
    fcode = [defclass[oo] for oo in fcode]
#     print(defclass)
#     print(fcode)
    return fpath, fcode

maindir = './动作识别-front-txt/动作正样本/组合9-24'
camera = 'front'

lsttest = os.listdir(maindir)
lsttest = sorted(lsttest)

afolder = lsttest[0]
sendp, sendact = maketestdata(afolder, defclass)
testdata = testtxt2tensor(sendp, camera, sendact)

for i in range(1,len(lsttest)):
    afolder = lsttest[i]
    sendp, sendact = maketestdata(afolder, defclass)
    testdata += testtxt2tensor(sendp, camera, sendact)

# ########## shuffle initial data ##################
for _ in range(50):
    np.random.shuffle(testdata)

torch.save(testdata, "ARNDatasetTestCTCV2.pt")

print('class: ', len(defclass)+1, 'testing data size: ', len(testdata))

print('finish load test data')
