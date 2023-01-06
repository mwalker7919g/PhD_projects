import cv2
import time
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from functions import *
from yumipy import YuMiRobot, YuMiState

def move_sample():
    
    # instructions for robot to pick sample, bring towards camera & rotate
    y = YuMiRobot()
    y.right.set_speed(YuMiRobot.get_v(150))
    y.right.goto_state(YuMiState([41.79, -143.36, 19.47, 15.26, 42.41, 89.57, -90.64]))
    y.right.set_speed(YuMiRobot.get_v(250))  
    y.right.goto_state(YuMiState([47.36, -143.36, 16.37, 18.78, 34.78, -6.62, -87.05]))
    y.right.goto_state(YuMiState([47.36, -143.36, 15.41, 19.41, 31.3, -3.28, -82.68]))

def record_video(model_type):

    # recording video to feed into inference model. Each model requires different video lengths
    cap = cv2.VideoCapture(0)
    start = time.time()
    if model_type == str('c'):
        capture_duration = 5

    elif model_type == str('r'):
        capture_duration = 20
    # codec
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('inference.avi', fourcc, 15, (640,480))
    i = 0
    out_path = "frames_inference"
    print(out_path)
    while (int(time.time() - start) < capture_duration) and i<30 :
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640,480))
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        frame_name = 'inference_'+str(i)+'.jpg'
        cv2.imwrite(os.path.join(out_path,frame_name), frame)
       
       # number of frames required for each CNN is different, so the video cuts at these thresholds
        if model_type == str('c'):
            i += 1
        elif model_type == str('r'):
            i += 2

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return out_path

# Convolutional Neural Network for Regression analysis of video

class CNN3D_Regression(nn.Module): 


    def __init__(self, t_dim=14, img_x=256 , img_y=342, drop_p=0, fc_hidden1=256, fc_hidden2=256):
        super(CNN3D_Regression, self).__init__()      
        # dimensions of video  
        self.t_dim = t_dim
        self.img_x = img_x  
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
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
                             self.fc_hidden1)  
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, 1)  # fully connected layer, output = one value
        
        
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

# Convolutional Neural Network for Classification

class CNN3D_Classification(nn.Module):

    def __init__(self, t_dim=30, img_x=256, img_y=342, drop_p=0, fc_hidden1=256, fc_hidden2=128, num_classes=4):
        super(CNN3D_Classification, self).__init__()  
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

def run_inference(out_path, model_type):
    # use CPU 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if model_type == str('c'):
        model = CNN3D_Classification()

    elif model_type == str('r'):
        model = CNN3D_Regression()




    path = out_path

    img_x = 256
    img_y = 342
    X = []

    # apply transformation to each image in inference folder
    for i in os.listdir(path):

        image = Image.open(os.path.join(path,i))
        image = image.convert('L')
        use_transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])

        if use_transform is not None:
            image = use_transform(image)

        X.append(image.squeeze_(0))
    
    # stack tensors along new dimension
    X = torch.stack(X,dim=0)
    X = X.unsqueeze_(0)  

    # load classification or regression model 
    if model_type == str('c'):

        model.load_state_dict(torch.load('3dcnn_epoch77.pth'))
    elif model_type == str('r'):
        model.load_state_dict(torch.load('reg_3dcnn_epoch1.pth'))

    print(model)
    model.eval()

    # apply model to inference frames
    output = model(X[None, ...])

    softmax_output = nn.Softmax(dim=1)
    output_softmax = softmax_output(output)
    print('output =',output)
    print('output softmax =', output_softmax)
    output_array = output.cpu().detach().numpy()


if __name__ == "__main__":
    model_type = str(input("Model to use Regression or Classifier? r/c = "))
    move_sample()
    out_path = record_video(model_type)
    run_inference(out_path, model_type)

