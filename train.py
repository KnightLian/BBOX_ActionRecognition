import time
import os
import matplotlib.pyplot as plt
from model import *

def training(indata, num_epochs):
    print("Training")
    total_step = len(indata)
    loss_count = []
    starttime = time.time()

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for index, (event, label) in enumerate(indata):
            optimizer.zero_grad()
            event = event.cuda()
            predict = model(event, None) #check to confirm sentence lenght
            batchs =predict.shape[1]
            label = label.cuda()
            preds_size = torch.IntTensor([predict.shape[0]]*batchs).cuda()
            label_size =  torch.IntTensor([label.shape[0]]*batchs).cuda()

            loss = lossfunc(predict, label, preds_size, label_size)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            processtime = time.time()

            if (index+1) % (int(len(indata)/10)) == 0: #bank data for graph
                loss_count.append(loss.item())

        if (epoch+1) % 10 == 0: #show loss at every 10 epoch
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}, Spend: {} sec'
                   .format(epoch+1, num_epochs, index+1, total_step, train_loss
                   ,round((processtime - starttime), 1)))

        if (epoch+1) % 100 == 0: #save the weights every 100 epoch
            if not os.path.exists("weights"):
                os.mkdir("weights")
            torch.save(model.state_dict(), f"weights/model_{epoch}.pth")

    # plot loss
    plt.figure('Training Loss')
    plt.plot(loss_count, label=f"Input Dims: {list(event.size())}")
    plt.legend()
    plt.savefig("training_loss.png")

    endtime = time.time()
    print('Time used:', round((endtime - starttime)/60/60, 2),'hrs')
    print('Final event size:', event.size())

    return 0

if __name__ == '__main__':
    # load training data
    pthfile = "data/ARNDatasetTrainSingleV2.pt"
    indata = torch.load(pthfile)

    # load model
    model = Model(416, 416, 1, 5, 1).cuda()
    print(model)

    # parameter
    num_epochs = 1000
    learning_rate = 0.001

    lossfunc = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0002)
    training(indata, num_epochs)
