import torch
import torch.nn as nn
import torch.nn.functional as F

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
