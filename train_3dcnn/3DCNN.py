import os
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import sys
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from torch.utils.data.sampler import SubsetRandomSampler
import shutil
from pathlib import Path
import itertools
import itertools
from itertools import product
import numpy as np

# use CPU for running
use_cuda = torch.cuda.is_available()                   
device = torch.device("cuda" if use_cuda else "cpu") 

# setting the random seeds
random_state=0
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def create_datasets(): # splitting data into train, validation and testing
   


    selected_frames_NL_split = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 45,46,47,48,49,50,52,53,54,55, 56,57,58]                              
    selected_frames= selected_frames_NL_split # choosing the frames of each video for training (chosen by observing liquid movement in videos)

    # detect device and use cpu
    use_cuda = torch.cuda.is_available()                   
    device = torch.device("cuda" if use_cuda else "cpu")   

    # file paths
    data_path = "lab_liqs_frames"         
    viscosity_cat_path = "labliqs.pkl"
    save_model_path = "./Conv3D_ckpt_class_test/"  

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    with open(viscosity_cat_path, 'rb') as f:
        visc_values = pickle.load(f)   

    # converting labels to categories
    le = LabelEncoder()
    le.fit(visc_values)
    print(visc_values)
    # give classes
    list(le.classes_)

    # convert categories using OneHotEncoder
    visc_cats = le.transform(visc_values).reshape(-1, 1)
    print(visc_cats)
    enc = OneHotEncoder()
    enc.fit(visc_cats)

    viscosities = []
    fnames = os.listdir(data_path)

    all_viscs = []
    
    # loop through folder
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        viscosities.append(f[(loc1 + 2): loc2])
        all_viscs.append(f)
    
    # get all videos (X data) and corresponding labels 
    all_X_list = all_viscs              
    all_y_list = labels2cat(le, viscosities)
    print(all_y_list)
    
    # % of training set is validation
    validation_size = 0.2

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.205, random_state=0)

    # transform images
    img_x, img_y = 256, 342
    
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

    

    # split train and test set	
    train_set, test_set = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)
    
    
   # split into training and validation 
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)

    # get train, valid and test loaders
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               num_workers=0)

    


    return train_loader, test_loader, valid_loader
    
class CNN3D(nn.Module):
    def __init__(self, t_dim=29, img_x=256 , img_y=342, drop_p=0, fc_hidden1=256, fc_hidden2=256, num_classes=5):
        # dimensions of video 
        super(CNN3D, self).__init__()        
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5,5,5), (3, 3, 3)  # 3-dimensional kernel size for convolutions
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # stride - degree of movement across each convolution step
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding        
        # output shapes from conv1 and conv2 layers
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        # ReLu activation layer adds non linearity 
        self.relu = nn.ReLU(inplace=True)
        # Drop random nodes to prevent overfitting
        self.drop = nn.Dropout3d(self.drop_p)
        # Shrink size of layers for computational time
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2*self.conv2_outshape[0]*self.conv2_outshape[1]*self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes 
        
        
    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)        
        return x
    
model = CNN3D()
print(model)

def train(model,batch_size, patience, epochs):
    
    # metric for model performance - Cross Entropy
    criterion = nn.CrossEntropyLoss() 
    # construct optimizer
    optimizer = torch.optim.Adam(model.parameters())
 
 	
    # set model to training mode	
    model.train()
    
    # lists to track train and valid loss
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    save_model_path = "./Conv3D_ckpt_labliquids/"  
    # counting total trained sample in one epoch
    N_count = 0   
    
    # no. epochs to wait until enfore early stopping
    patience = 1
    # perform early stopping (prevent overfitting)
    early_stopping = EarlyStopping(patience=patience, verbose = True)
    
    for epoch in range(epochs):
        
        for batch_idx, (X, y) in enumerate(train_loader):
             
            epoch_subtract = 0
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            N_count += X.size(0)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            train_losses.append(loss.item()) 
            loss.backward()
            optimizer.step()
    
        # validate the model
        model.eval()
        for (X, y) in valid_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            output = model(X)
            loss = criterion(output, y)
            valid_losses.append(loss.item())                 
                   
        # av loss over epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
    
        
        print(print_msg)
        
        # clearing list to track next epoch
        train_losses = []
        valid_losses = []
        
        # early stopping to prevent overfitting model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            epoch_subtract += 1
            print("Early stopping")
            break

    summation = epoch  - patience
    print(summation)

    # save models
    torch.save(model.state_dict(), os.path.join(save_model_path, 'lr_'+str(learning_rate)+'_batch_size_'+str(batch_size)+'_3dcnn_epoch{}.pth'.format(summation)))  # save spatial_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, str(batch_size)+'_3dcnn_optimizer_epoch{}.pth'.format((summation))))      # save optimizer
    print("Epoch {} model saved!".format(summation))
    
    
    return  model, avg_train_losses, avg_valid_losses
    

    
def run_testing(model): # loop through models and run on test data to optimise performance

    # put models into folder
    source_dir = 'Lab_liquids_optimisers'
    target_dir = 'Lab_liquids_models'
    for lrbs in os.listdir('Lab_liquids_optimisers'):
        if str('lr') in lrbs:

            shutil.move(os.path.join(source_dir,lrbs), target_dir)

    target_dir = 'Lab_liquids_models'

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # lists for tracking model performances
    model_parameter = []
    accuracies = []
        
    for model_permutation in os.listdir(target_dir): 

        model = CNN3D()

        # load each model
        model.load_state_dict(torch.load('./Lab_liquids_models/'+str(model_permutation)))
        
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(5))
        class_total = list(0. for i in range(5))
        
        # set model to evaluation mode
        model.eval() 
        # create lists for comparing true and predicted values
        true_values = [] 
        predicted_values = []
        saved_list = []
        num_tests = 46
	
        for batch,(data, target) in enumerate(test_loader):

            target1 = target.data[:,0]
            target1_array = target1.cpu().detach().numpy()

            # Computing predicted outputs
            output = model(data)
            # get loss
            loss = criterion(output, target1)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # Output probability --> predicted class
            _, pred = torch.max(output, 1)

            prediction_integer = pred.tolist()
           
            # was prediction correct?
            correct = np.squeeze(pred.eq(target1.data.view_as(pred)))
           
            # get accuracy for each class in range of test set
            for i in range(num_tests):
                label = target1.data[i]
                label_integer = label.item()
                true_values.append(label_integer)
                class_correct[label] += correct[i].item()
                class_total[label] += 1

            pred_integer = pred.view(pred.numel()).numpy()
            predicted_values.append(pred_integer)

            # give test loss
            test_loss = test_loss/len(test_loader.dataset)
            print('Test Loss: {:.6f}\n'.format(test_loss))
            
            for i in range(5):
                if class_total[i]>0:
                    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
            print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))

            overall_accuracy = 100 * np.sum(class_correct) / np.sum(class_total)
            model_parameter.append(model_permutation)
            accuracies.append(overall_accuracy)

            # get results of model accuracy with each permutation 
            
            results_table = pd.DataFrame(np.c_[model_parameter,accuracies],columns = ["-","-"])

    print('----model parameters----             ---- accuracy (%) -----')
    print(results_table)
    

if __name__=="__main__":
    model = CNN3D()

    # hyper param optimisation
    epochs = 300
    patience = 50
    batch_sizes = [2,4,8,16,32]
    learning_rates = [0.1,0.01,0.001,0.0001]

    bs_lr_list = list(itertools.product(batch_sizes, learning_rates))
    dct = {'first': batch_sizes, 'second': learning_rates}
    dct_list = list(product(*dct.values()))

    # run functions on each paramter permutation
    for (batch_size, learning_rate) in (dct_list):

       train_loader, test_loader, valid_loader = create_datasets()
       model, train_loss, valid_loss  = train(model,batch_size, patience, epochs)

    run_testing(model)
    
    
