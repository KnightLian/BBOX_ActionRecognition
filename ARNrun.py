import os
import torch
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

class BiLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nLayer):
        super(BiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(  input_size=nIn,
                                hidden_size=nHidden,
                                num_layers=nLayer,            
                                dropout=0,
                                bidirectional=True,
                                batch_first=True 
                             )  
    def forward(self, x, hidden):
        x, (h,c) = self.bilstm(x, hidden)
        return x, (h,c)  

class Model(nn.Module):
    def __init__(self, nIn, nHidden, nLayer, nOut, CNNinchannel):
        super(Model, self).__init__()
        
        self.conv1 = nn.Sequential(                                                                                                                                                                                                                                                                                                                                 
                                   nn.Conv2d(   in_channels=1*CNNinchannel, 
                                                 out_channels=8*CNNinchannel,
                                                 kernel_size=(3, 1),   
                                                 stride=(1, 1),  
                                                 padding=(1, 0) 
                                             )  
                                   ,nn.BatchNorm2d(8)
                                   ,nn.LeakyReLU(0.1, inplace=True)                                 
                                   ,nn.Conv2d(   in_channels=8*CNNinchannel, 
                                                 out_channels=8*CNNinchannel,
                                                 kernel_size=(3, 1),   
                                                 stride=(1, 1),  
                                                 padding=(1, 0) 
                                             )  
                                   ,nn.BatchNorm2d(8)
                                   ,nn.LeakyReLU(0.1, inplace=True)              
                                   ,nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))                
            
                                   ,nn.Conv2d(   in_channels=8*CNNinchannel, 
                                                 out_channels=16*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(16)
                                   ,nn.LeakyReLU(0.1, inplace=True)                                 
                                   ,nn.Conv2d(   in_channels=16*CNNinchannel, 
                                                 out_channels=16*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(16)
                                   ,nn.LeakyReLU(0.1, inplace=True) 
                                   ,nn.Conv2d(   in_channels=16*CNNinchannel, 
                                                 out_channels=16*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(16)
                                   ,nn.LeakyReLU(0.1, inplace=True)             
                                   ,nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                  
                           
                                   ,nn.Conv2d(   in_channels=16*CNNinchannel, 
                                                 out_channels=32*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(32)
                                   ,nn.LeakyReLU(0.1, inplace=True)                                 
                                   ,nn.Conv2d(   in_channels=32*CNNinchannel, 
                                                 out_channels=32*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(32)
                                   ,nn.LeakyReLU(0.1, inplace=True)                                                    
                                   ,nn.Conv2d(   in_channels=32*CNNinchannel, 
                                                 out_channels=32*CNNinchannel,
                                                 kernel_size=(3, 3),   
                                                 stride=(1, 1),  
                                                 padding=(1, 1) 
                                             )  
                                   ,nn.BatchNorm2d(32)
                                   ,nn.LeakyReLU(0.1, inplace=True)            
                                   ,nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                                                                                  
                                  )          
                   
        self.rnn1 = BiLSTM(nIn, nHidden, nLayer) 
    
        self.fc1 = nn.Sequential( nn.Linear(nHidden*2, 512) 
                                 ,nn.LeakyReLU(0.1, inplace=True)  
                               
                                 ,nn.Dropout(p=0.5)
                                 ,nn.Linear(512, 512) 
                                 ,nn.LeakyReLU(0.1, inplace=True)  
                                 
                                 ,nn.Dropout(p=0.5) 
                                 
                                 ,nn.Linear(512, 256) 
                                 ,nn.LeakyReLU(0.1, inplace=True)           
                                 
                                 ,nn.Linear(256, nOut)
                                 ,nn.LogSoftmax(dim=2) 
                                )   
    
    def forward(self, x, hidden):   

#         print('Indata：', x.size()) 

        x = self.conv1(x) 
#         print('CNNout:', x.size())      
        
        x = x.permute(0, 2, 1, 3)
#         print('Transpose:', x.size())

        x = torch.flatten(x, 2, 3) 
#         print('Reshape:', x.size()) 
        
        x, (h,c) = self.rnn1(x, hidden)    
#         print('LSTMout:', x.size())  

        x = x.permute(1, 0, 2)
#         print('Transpose:', x.size())

        x = self.fc1(x)  
#         print('Linearout:', x.size())   
    
        return x    
    
if __name__ == "__main__": 
    # load training and validating data
    pthfile = "CreateDataSet/ARNDatasetTrainSingleV2.pt" 
    indata = torch.load(pthfile)
    print('finish load traning data')
    
    # load model
    model = Model(416, 416, 1, 5, 1).cuda() 
    print(model) 

    # parameter
    num_epochs =  1000 #定义跑多少代
    learning_rate = 0.001 #学习率

    lossfunc = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0002)

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

            # if (index+1) % (int(len(indata)/10)) == 0: #bank data for graph
            #     loss_count.append(loss.item()) 

        if (epoch+1) % 10 == 0: #show loss at every epoch
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}, Spend: {} sec' 
                   .format(epoch+1, num_epochs, index+1, total_step, train_loss
                   ,round((processtime - starttime), 1)))   

        if (epoch+1) % 100 == 0: #save the model every 50 epoch
            if not os.path.exists("model"):
                os.mkdir("model")
            torch.save(model.state_dict(), f"model/model_{epoch}.pth")      

    # plot loss
    plt.figure('CNNLSTM_Loss')
    plt.plot(loss_count,label='Loss.item()')
    plt.legend()
    plt.show()

    endtime = time.time()
    print('time used:', round((endtime - starttime)/60/60, 2),'hrs')
    print('Final event size:', event.size()) 
    
