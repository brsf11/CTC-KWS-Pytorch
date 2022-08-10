import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import json


class KWSDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.framedata = []
        self.frametag = []
        self.framelen = []
        for file in self.file_list:
            jsonfile = open(self.path+file,'r')
            jsondata = json.loads(jsonfile.read())
            for data in jsondata:
                self.framedata.append(torch.tensor(data['data']))
                self.frametag.append(torch.tensor(data['tag']))
                self.framelen.append(torch.tensor(data['len']))

    def __getitem__(self, index):
        data = self.framedata[index]
        tag = self.frametag[index]
        length = self.framelen[index]
        return data,tag,length

    def __len__(self):
        return len(self.frametag)

training_data = KWSDataset("/home/brsf11/Hdd/ML/Dataset/MobvoiHotwords/output/")
        

batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)

print(f"length: {len(train_dataloader.dataset)}")
for X,y,l in train_dataloader:
    print(f"Shape of X [N, C, H]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"Shape of len: {l.shape} {l.dtype}")
    break



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,x):
        logits = self.linear_relu_stack(x)
        return logits

model = NerualNetwork().to(device)
print(model)

loss_fn = nn.CTCLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

input_lengths = torch.full(size=(batch_size,), fill_value=298, dtype=torch.long)

def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y,l) in enumerate(dataloader):
        X,y,l = X.to(device),y.to(device),l.to(device)

        pred = model(X).permute(1,0,2)
        loss = loss_fn(pred,y,input_lengths,l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 8 == 0:
            loss, current = loss.item(),batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# def test(dataloader,model,loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss,correct = 0,0
#     with torch.no_grad():
#         for X,y in dataloader:
#             X,y = X.to(device),y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred,y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
print("Done!")
