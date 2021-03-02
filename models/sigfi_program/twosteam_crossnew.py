# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:57:26 2020

@author: Administrator
"""

import numpy  as np
import scipy.io as sio
from two_stream18 import CNN,BasicBlock,BasicBlock_p
from two_resnet50 import CNN50,Bottleneck

data_type = 'lab'
num_class=150
if num_class==276:
    if data_type == 'home':
        path1='dataset_home_276.mat'
        home276 = sio.loadmat(path1)
    elif data_type == 'lab':
        path1='dataset_lab_276_dl.mat'
        home276 = sio.loadmat(path1)
    else:
        path2='dataset_lab_276_dl.mat'
        path1='dataset_home_276.mat'   
        lab276 = sio.loadmat(path2)
        home276 = sio.loadmat(path1)
    print(home276.keys())
elif num_class==150:
    path1='dataset_lab_150.mat'
    home150 = sio.loadmat(path1)
    print(home150.keys())

import torch
from torch import nn
from torch.utils.data import Dataset,TensorDataset
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from collections import  Counter
from resnet_tt18 import ResNet,BasicBlock,resnet18,resnet50,resnext50_32x4d
from model_resnet import ResidualNet
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet50" #"resnet50"

# Number of classes in the dataset
num_classes = num_class

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 500

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False #
best_acc_list=[]
best_acc_sum=0

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}  


def max (data,n):
        data_vec1, data_val, data_vec2 = np.linalg.svd(data, full_matrices=False)
        data_val_any = np.diag(data_val[: n])
        mat_new_any = data_vec1[:, 0:n]@data_val_any@data_vec2[0:n, :]
        return mat_new_any#Image.fromarray(mat_new_any).show()
def svd_r(X,n):
    svd_result=[]
    for i in range(len(X)):
        image1=X[i] 
        image_svd =np.zeros_like(image1)
        image_svd[:,:,0]=max(image1[:,:,0],n)
        image_svd[:,:,1]=max(image1[:,:,1],n)
        image_svd[:,:,2]=max(image1[:,:,2],n)
        svd_result.append(image_svd)
    svd_result=np.array(svd_result)
    return svd_result 
#%%
if num_class==150:
    def amp_phs():
        P1=np.angle(home150['csi1'],deg=True).astype(int)
        X1 =np.concatenate((abs(home150['csi1']),abs(P1)),axis = 1)
        P2=np.angle(home150['csi2'],deg=True).astype(int)
        X2 =np.concatenate((abs(home150['csi2']),abs(P2)),axis = 1)
        P3=np.angle(home150['csi3'],deg=True).astype(int)
        X3 =np.concatenate((abs(home150['csi3']),abs(P3)),axis = 1)
        P4=np.angle(home150['csi4'],deg=True).astype(int)
        X4 =np.concatenate((abs(home150['csi4']),abs(P4)),axis = 1)
        P5=np.angle(home150['csi5'],deg=True).astype(int)
        X5 =np.concatenate((abs(home150['csi5']),abs(P5)),axis = 1)
        r1=np.concatenate((X1.transpose(3,0,1,2),X2.transpose(3,0,1,2),X3.transpose(3,0,1,2),X4.transpose(3,0,1,2),X5.transpose(3,0,1,2)),axis = 0)
        r1[:,:,:30,:]=svd_r(r1[:,:,:30,:],15)
        r1[:,:,30:,:]=svd_r(r1[:,:,30:,:],15)
        diff0=r1[:,:-1,:,:]
        diff1=r1[:,1:,:,:]
        diff=diff1-diff0
        # diff[:,:,:30,:]=svd_r(diff[:,:,:30,:],20)
        # diff[:,:,30:,:]=svd_r(diff[:,:,30:,:],20)
        X =np.concatenate((diff0,diff),axis = 2)
        return X
    X=amp_phs()
    Y = home150['label']
    Y =Y[:,0]-1
    for i in range(len(Y)):
        if Y[i]>128:
            Y[i]=Y[i]-128
elif num_class==276:
    if data_type == 'home':
        # X2 =20*np.log10(np.abs(home276['csid_home'])+1)
        X=np.abs(home276['csid_home'])
        P=np.angle(home276['csid_home'],deg=True)#+ 3.141592653589793#%%.astype(int) 
        Y1 = home276['label_home']
    elif data_type == 'lab':
        X=np.abs(home276['csid_lab'])
        P=np.angle(home276['csid_lab'],deg=True)
        Y1 = home276['label_lab']
    else:
        X_lab=np.abs(lab276['csid_lab'])
        P_lab=np.angle(lab276['csid_lab'],deg=True)
        Y_lab = lab276['label_lab']
        # Y_lab =Y_lab[:,0]-1  
        X_home=np.abs(home276['csid_home'])
        P_home=np.angle(home276['csid_home'],deg=True)#+ 3.141592653589793#%%.astype(int) 
        Y_home = home276['label_home']

        X=np.concatenate((X_lab,X_home),axis = 3)      
        P=np.concatenate((P_lab,P_home),axis = 3)       
        # Y=np.concatenate((Y_lab,Y_home),axis = 0)
       
        Y1=np.concatenate((Y_lab,Y_home),axis = 0)
 #%%      
    Y =Y1[:,0]-1
     
    #X1 =np.concatenate((abs(home276['csiu_home']),abs(home276['csid_home'])),axis = 1)
    # W=np.unwrap( P,discont=np.pi,axis=1)#+8.28318531#+24  unwrap[unwrap>0].shape
    
    # if abs(np.min(W))>(np.max(W)):
    #     W= W-(np.max(W)+ np.min(W))/2
    # else:
    #     W= W+(np.max(W)+ np.min(W))/2 
    r1=np.concatenate((X,P),axis = 1).transpose(3,0,1,2)
    # r2=np.concatenate((X1,W),axis = 1).transpose(3,0,1,2)
    # r1[:,:,:30,:]=svd_r(r1[:,:,:30,:],20)
    # r1[:,:,30:,:]=svd_r(r1[:,:,30:,:],20)
    diff0=r1[:,:-1,:,:]
    diff1=r1[:,1:,:,:]
    diff=diff1-diff0
    diff[:,:,:30,:]=svd_r(diff[:,:,:30,:],20)
    diff[:,:,30:,:]=svd_r(diff[:,:,30:,:],20)
    X =np.concatenate((r1[:,:-1,:,:],diff),axis = 2)

def normalization(data): 
    _range = np.max(data)- np.min(data)
    return (data - np.min(data)) / _range
X[:,:,:30,:]=normalization(X[:,:,:30,:])
X[:,:,30:60,:]=normalization(X[:,:,30:60,:])  
X[:,:,60:90,:]=normalization(X[:,:,60:90,:])
X[:,:,90:,:]=normalization(X[:,:,90:,:])  


#%%
# import matplotlib.pyplot as plt  
# data1=diff0[:1000,:,:,:].flatten()
# data2=diff[:1000,:,:,:].flatten()
# #%%
# hist1,_ = np.histogram(data1,bins=1000)
# hist2,_ = np.histogram(data2,bins=1000)
# #%%
# # num_bins =1000  
# # n, bins, patches = plt.hist(data1, num_bins,  facecolor='blue')  
# #     # add a 'best fit' line  
# # # y = mlab.normpdf(bins, mu, sigma)  
# plt.plot(hist2)  
# plt.xlabel('Smarts')  
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  


'''
#%%
#SVD image show
r2=np.zeros_like(r1)
r2[:,:,:30,:]=svd_r(r1[:,:,:30,:],20)
r2[:,:,30:,:]=svd_r(r1[:,:,30:,:],20)
diff_o=diff1-diff0
r2[:,:,:30,:]=normalization(r2[:,:,:30,:])
r2[:,:,30:60,:]=normalization(r2[:,:,30:60,:])  
diff_o[:,:,:30,:]=normalization(diff_o[:,:,:30,:])
diff_o[:,:,30:60,:]=normalization(diff_o[:,:,30:60,:])  
def ToRGB(data):
    rgbdata=[]
    data=data.transpose(0,3,1,2)
    for i in range(len(data)):     
        # data[i][:,:,:30] = normalization(data[i][:,:,:30])
        # data[i][:,:,30:] = normalization(data[i][:,:,30:])#负数 data[i][data[i]<0]
#        print(type(x_data))
        img_2 = transforms.ToPILImage()(torch.from_numpy(data[i].astype(np.float32))).convert('RGB')#np.float32默认float64报错
        rgbdata.append(img_2)
    return rgbdata   

trainx_a=ToRGB(X[:,:,:60,:])
svdx_a=ToRGB(r2[:,:,:60,:])
trainx_p=ToRGB(diff_o[:,:,:60,:])
svdx_p=ToRGB(X[:,:,60:,:]) 


plt.imshow(trainx_a[0])
plt.show()
plt.imshow(svdx_a[0])
plt.show()
plt.imshow(trainx_p[0])
plt.show()
plt.imshow(svdx_p[0])
plt.show()'''
#%%
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ToRGB(data):
    rgbdata=[]
    data=data.transpose(0,3,1,2)
    for i in range(len(data)):     
        # data[i][:,:,:30] = normalization(data[i][:,:,:30])
        # data[i][:,:,30:] = normalization(data[i][:,:,30:])#负数 data[i][data[i]<0]
#        print(type(x_data))
        img_2 = transforms.ToPILImage()(torch.from_numpy(data[i].astype(np.float32))).convert('RGB')#np.float32默认float64报错
        rgbdata.append(img_2)
    return rgbdata
class TrainDataset(Dataset):
    def __init__(self,x_a,x_p,y,transform=None):
        # assert data_tensor.size(0) ==  target_tensor.size(0)
        self.transform = transform
        
        self.y_data = torch.from_numpy(y.astype(np.int32)).type(torch.LongTensor)#[:,0]
        
        self.xa = x_a
        self.xp = x_p
    def __getitem__(self, index):
         sample_a=self.xa[index]
         sample_p=self.xp[index]
         if self.transform:
            sample_a = self.transform(sample_a)
            sample_p = self.transform(sample_p)
         return sample_a,sample_p,self.y_data[index]
    def __len__(self):
        return len(self.xa)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
svc_acc_sum=[]
def svc(traindata, trainlabel, testdata, testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1,kernel="rbf",cache_size=3000)
    
    # svcClf =  RandomForestClassifier(criterion='gini',
    #                             n_estimators=25, 
    #                             random_state=1,
    #                             n_jobs=2)
    # svcClf =  KNeighborsClassifier(n_neighbors=5, 
    #                        p=2, 
    #                        metric='minkowski')
    svcClf.fit(traindata, trainlabel)
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("svm Accuracy:", accuracy)
    svc_acc_sum.append(accuracy)
# svc(train_features,  trianlabels_list, conv_features,labels_list) 
#%%
import torch.nn.functional as F
class My_dataset(Dataset):
    def __init__(self,feat,labels):
        self.conv_feat = feat
        self.labels = labels
    
    def __len__(self):
        return len(self.conv_feat)
    
    def __getitem__(self,idx):
        return self.conv_feat[idx],self.labels[idx]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.drop_lay = nn.Dropout(0.5)
        # self.conv2_drop = nn.Dropout2d()#nn.Dropout2d
        self.fc1 = nn.Linear(num_class, 1024)
        self.fc2 = nn.Linear(1024,2048)        
        self.fc3 = nn.Linear(2048, num_class)
        # self.fc1 = nn.Linear(552, 2048)
        # self.fc2 = nn.Linear(2048,1024)        
        # self.fc3 = nn.Linear(1024, 276)
    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # x = self.drop_lay(x)
        # x = F.dropout(x,training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

train_accuracy,val_accuracy =[],[]
def fit_numpy(epoch,model,optimizer,data_loader,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
      
    criterion1 = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx , data_d  in enumerate(data_loader):       
        inputs, labels = data_d 
        inputs = inputs.to(device)
        labels = labels.to(device)
        if phase == 'training':
            optimizer.zero_grad()
        # data = data.view(data.size(0), -1)
        output = model(inputs)
        loss = criterion1(output,labels)
        _, preds = torch.max(output, 1)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)          
        # running_loss += F.cross_entropy(output,target,size_average=False).item()
        # preds = output.data.max(dim=1,keepdim=True)[1]
        # running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = running_corrects.double() / len(data_loader.dataset)
    # loss = running_loss/len(data_loader.dataset)
    # accuracy = 100. * running_correct/len(data_loader.dataset)
    
    # print(f'{phase} loss is {epoch_loss:{5}.{2}} and {phase} accuracy is {running_corrects}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss,accuracy

def fc_net(train_features,  trianlabels_list, conv_features,labels_list):
    train_feat_dataset = My_dataset(train_features,  trianlabels_list)
    val_feat_dataset = My_dataset(conv_features,labels_list)
    train_feat_loader = DataLoader(train_feat_dataset,batch_size=32,shuffle=True)
    val_feat_loader = DataLoader(val_feat_dataset,batch_size=32,shuffle=False)
    vgg = Net( )  
    vgg = vgg.cuda()
    optimizer = optim.SGD(vgg.parameters(),lr=0.0001,momentum=0.5)
    # train_losses , train_accuracy = [],[]
    # val_losses , val_accuracy = [],[]
    for epoch in range(1,300):
        if epoch == 100:
            set_learning_rate(optimizer, 0.0005) 
        elif epoch == 200:
            set_learning_rate(optimizer, 0.00025) 
        elif epoch == 300:
            set_learning_rate(optimizer, 0.00015) 
        epoch_loss, epoch_accuracy = fit_numpy(epoch,vgg,optimizer,train_feat_loader,phase='training')
        val_epoch_loss , val_epoch_accuracy = fit_numpy(epoch,vgg,optimizer,val_feat_loader,phase='validation')
        # train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        # val_losses.append(val_epoch_loss)
        
        if epoch == 299:
            val_accuracy.append(val_epoch_accuracy.item())
        if epoch%100 == 99:
            print(f'train accuracy is {epoch_accuracy.item()} val accuracy is {val_epoch_accuracy.item():{10}.{4}}')
#%%
conv_features1 = []
labels_list1 = []
train_features1 = []
trianlabels_list1 = []

conv_features = []
labels_list = []
train_features = []
trianlabels_list = []
def train_model(model, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    valp_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc_history = []
    loss_history  = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        # Each epoch has a training and validation phase
        if epoch == 370:
            set_learning_rate(optimizer, 0.00075) 
        if epoch == 400:
            set_learning_rate(optimizer, 0.0005) 
        if epoch == 470:
            set_learning_rate(optimizer, 0.00025) 
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_corrects_p = 0
            # print('is_inception',is_inception)
            # Iterate over data.
            for _,data_d  in enumerate(phase_dic[phase]):
            
                inputs,p_inputs, labels = data_d                               
                inputs = inputs.to(device)
                p_inputs = p_inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs,p_inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # input_tensor = torch.ones_like(inputs).to(device)
                        # outputs = model(inputs,p_inputs)
                        outputs,_= model(inputs,p_inputs)
                        _,outputs_p= model(inputs,p_inputs)
                        # out_all=torch.cat([outputs,outputs_p],dim=1)
                        out_all=torch.mul(outputs, outputs_p)
                        
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(outputs_p, labels)   
                        # loss = criterion(outputs, labels)
                        loss =loss1+loss2
                    _, preds = torch.max(outputs, 1)
                    _, preds_p = torch.max(outputs_p, 1)
                    # print('preds',preds.data)
                    # print('labes',labels.data)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if epoch ==(num_epochs-1) and phase == 'train':
                    # if phase == 'train' :#and epoch_acc > best_acc:
                     train_features1.extend(out_all.data.cpu().numpy())
                     trianlabels_list1.extend(labels.data.cpu().numpy())
                if epoch ==(num_epochs-1) and phase == 'val':
                     conv_features1.extend(out_all.data.cpu().numpy())
                     labels_list1.extend(labels.data.cpu().numpy())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_p += torch.sum(preds_p == labels.data)

            epoch_loss = running_loss / len(phase_dic[phase].dataset)
            epoch_acc = running_corrects.double() / len(phase_dic[phase].dataset)
            epoch_acc_p = running_corrects_p.double() / len(phase_dic[phase].dataset)
            if epoch % 100 == 99:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print('{}  pAcc: {:.4f}'.format(phase, epoch_acc_p))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'train' :#and epoch_acc > best_acc:
                train_acc_history.append(epoch_acc) 
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                valp_acc_history.append(epoch_acc_p)
                loss_history.append(epoch_loss)
                last_acc=epoch_acc
                last_acc_p=epoch_acc_p
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history,last_acc,last_acc_p,train_acc_history,loss_history,valp_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        new_file  = "new_44.pth"
        model_ft = CNN(BasicBlock, BasicBlock_p, [2, 2, 2, 2])
        pretrained_dict=torch.load(new_file)
        model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
        model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
        model_ft.load_state_dict(model_dict)
        set_parameter_requires_grad(model_ft, feature_extract)
#        ct = 0
#        for child in model_ft.children():
#           ct += 1
#           print("child",child)
#           if ct < 6:
#               for param in child.parameters():
#                   param.requires_grad = False 
        #提取fc层中固定的参数  
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet50":
        """ Resnet18
        """
        # model_ft = models.resnet18(pretrained=use_pretrained)
        new_file  = "two_50.pth"
        model_ft = CNN50(Bottleneck, [3, 4, 6, 3])
        pretrained_dict=torch.load(new_file)
        model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
        model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
        model_ft.load_state_dict(model_dict)
        set_parameter_requires_grad(model_ft, feature_extract)
#        ct = 0
#        for child in model_ft.children():
#           ct += 1
#           print("child",child)
#           if ct < 6:
#               for param in child.parameters():
#                   param.requires_grad = False 
        #提取fc层中固定的参数  
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "twostream":
        # model_ft = ResidualNet( 'ImageNet', 18, 1000, 'CBAM' )
        # new_file  = "resnet18-5c106cde.pth"
        pretrained_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        # model_ft = ResNet(BasicBlock,  [2, 2, 2, 2])
        model_ft = CNN(BasicBlock, BasicBlock_p, [2, 2, 2, 2])
        # pretrained_dict=torch.load(new_file)
        model_dict = model_ft.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
        model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
        model_ft.load_state_dict(model_dict)
        set_parameter_requires_grad(model_ft, feature_extract)
        #提取fc层中固定的参数  
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)
#%%
# model_dict = model_ft.state_dict()
# o=[]
# for k, v in model_dict.items() :
#     if(k=='layer3_a.1.bn1.weight'):
#         o.append(v)
#     if(k=='layer3_p.1.bn1.weight'):
#         o.append(v)  
# print(o[0]==o[1])
#%%
ohist = []#val
phist = []#pval
shist = []#train
lhist = []#loss
last_acc_list= []
last_acc_p_list= []

from sklearn.model_selection import StratifiedKFold#StratifiedShuffleSplit
# skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=None)
skf = StratifiedKFold(n_splits=5,shuffle=True)
j=0
for train_index , test_index in skf.split(X, Y):
    # print("%s %s" % (train_index , test_index))
    # if j==1:
    #     break
    # j=j+1
    train_x, train_y = X[train_index], Y[train_index]
    test_x, test_y = X[test_index], Y[test_index]
    trainx_a=ToRGB(train_x[:,:,:60,:])
    testx_a=ToRGB(test_x[:,:,:60,:])
    trainx_p=ToRGB(train_x[:,:,60:,:])
    testx_p=ToRGB(test_x[:,:,60:,:])
    # print(set(test_y))
    # print(Counter(test_y))
    # print(len(set(test_y)))
    # print(train_x.shape)#(2208,200, 30, 3 )
    # print(X.transpose(3,0,1,2)[0]==X[:,:,:,0])
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
   
    y_dic={'train':train_y, 'val':test_y}
    train_datasets = TrainDataset(x_a=trainx_a,x_p=trainx_p,y=y_dic['train'],transform =data_transforms['train'])
    val_datasets = TrainDataset(x_a=testx_a,x_p=testx_p,y=y_dic['val'],transform =data_transforms['val'])                  
    train_dataloaders =  torch.utils.data.DataLoader(train_datasets, batch_size=16,shuffle=True,num_workers=16)            
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=16,shuffle=False,num_workers=16)
                                  
    i=0
    def imshow(inp,cmap=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp,cmap)

    for train_set in val_dataloaders:
    
        fea, fea1,classes =train_set
    
        imshow(fea[1])
        imshow(fea1[1])
        # print(fea.size())
        # print(classes)
        # print(fea1.size())
  
        if i == 3:
            break
        else:
            i=i+1 
    # Initialize the model for this run
    phase_dic={'train':train_dataloaders, 'val':val_dataloaders}
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    # print(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(params_to_update, lr=0.0001)
# Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

# Train and evaluate
    model_ft, hist,last_acc,last_acc_p,scratch_hist,loss_hist,hist_p = train_model(model_ft,  criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    print(last_acc.item())
    ohist = [h.cpu().numpy() for h in hist]
    phist = [h.cpu().numpy() for h in hist_p]
    shist = [h.cpu().numpy() for h in scratch_hist]
    lhist = [h for h in loss_hist]
    conv_features = np.concatenate([[feat] for feat in conv_features1])
    train_features = np.concatenate([[feat] for feat in train_features1])
    labels_list = np.array(labels_list1)
    trianlabels_list =np.array(trianlabels_list1) 
    last_acc_list.append(last_acc.item())
    last_acc_p_list.append(last_acc_p.item())
    svc(train_features,  trianlabels_list, conv_features,labels_list) 
    fc_net(train_features,  trianlabels_list, conv_features,labels_list)
    #%%
avr_svcacc=  np.mean(svc_acc_sum)    
avr_acc= np.mean(last_acc_list)  #last_acc_sum/5 
fc_acc=np.mean(val_accuracy)
fusion_acc =np.mean(last_acc_p_list)
print('fc_acc',fc_acc) 
print('avr_acc',avr_acc)
print('svc_acc',avr_svcacc)
print('fusionavr_acc',fusion_acc)
#%%
'''
for i in range(len(testx)):
    plt.imshow(testx[i])
    plt.show()

'''

#%%
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="val")
plt.plot(range(1,num_epochs+1),phist,label="pval")
plt.plot(range(1,num_epochs+1),shist,label="train") 
plt.ylim((0,1.)) #坐标轴的范围
# plt.xticks(np.arange(1, num_epochs+1, 1.0))#刻度标签
plt.legend()
    
plt.figure()

plt.plot(range(1,num_epochs+1),lhist,label="loss")
plt.xlabel("Training Epochs")
plt.ylabel("loss")
plt.show()


