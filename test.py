import numpy as np
from model import *

def countscore(result, predict):
    correctact = 0.0
    for oneact in predict:
        if oneact in result:
            correctact+=1
    thisvideoscore = 100*correctact/len(result)
    return thisvideoscore

def calaccuracy(inputdata):
    allvideoactper = 0.0
    model.eval()
    with torch.no_grad():
        for index, (event, label) in enumerate(inputdata):
            event = event.cuda()
            predict = model(event, None) #check to confirm sentence lenght
            _, predict = predict.max(2)
            predict = predict.squeeze()
            predict = predict.contiguous()
    #         predict = torch.transpose(predict, 0, 1)  #keep if batch not = 1
            predict = predict.cpu().data.numpy()
    #         print("predictions at all frames: \n" predict)
            predict = predict[predict != 0]
            result = label.cpu().data.numpy()
#             print('predict before: ', predict, ' \t \t GT: ', result)
            predict = np.unique(predict)
            result = np.unique(result)
#             print('predict after: ', predict, ' \t \t GT: ', result)
            if  len(predict) <= len(result):
                thisvideoscore = countscore(result, predict)
            else:
                thisvideoscore = countscore(predict, result)
    #         if thisvideoscore<100:
    #             print('Pred:', predict, 'vs GT:', result, ', this video score: {:.0f}%' .format(thisvideoscore))
            allvideoactper += thisvideoscore
    return allvideoactper/len(inputdata)

if __name__ == '__main__':
    # load training, validation and testing data
    pthfile = "data/ARNDatasetTrainSingleV2.pt"
    indata = torch.load(pthfile)
    pthfile = "data/ARNDatasetValSingleV2.pt"
    valdata = torch.load(pthfile)
    pthfile = "data/ARNDatasetTestCTCV2.pt"
    testdata = torch.load(pthfile)

    # load model
    model = Model(416, 416, 1, 5, 1).cuda()
#     print(model)

    #load saved model file
    whichepoch = 699 # which epoch model to input
    pthfile = f"weights/model_{whichepoch}.pth"
    checkpoint = torch.load(pthfile)
    model.load_state_dict(checkpoint)

    accuracy = calaccuracy(indata)
    print('Training Accuracy: {:.2f}%' .format(accuracy))
    accuracy = calaccuracy(valdata)
    print('Validation Accuracy: {:.2f}%' .format(accuracy))
    accuracy = calaccuracy(testdata)
    print('Testing Accuracy: {:.2f}%' .format(accuracy))  
