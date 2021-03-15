#!/usr/bin/env python3.7
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset_VAE import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device = " + str(device))

class AutoEncoder(nn.Module):

 
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_conv_AutoEncoder")
    

  
    def __init__(self):
        super(AutoEncoder, self).__init__()
         
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)

        self.upconv3e = nn.ConvTranspose2d(32, 32, 4, stride = 2, padding=1)

        self.conv4 =  nn.Conv2d(32, 16, 2, stride=1, padding=1)

        self.upconv4e = nn.ConvTranspose2d(16,16,4, stride = 2, padding=1)

        self.conv_Im_z1 = nn.Conv2d(16, 1, 2, stride = 1, padding=1)
        self.conv_logvar = nn.Conv2d(16, 1, 2, stride = 1, padding=1)

        self.fc1d = nn.Linear(16 * 16, 8 * 8 * 2)
        self.fc2d = nn.Linear(2 * 8 * 8, 8 * 8)
        self.fc3d = nn.Linear(8 * 8, 7)
       

        
        self.fc1e = nn.Linear( 7 , 8 * 8)
        self.fc2e = nn.Linear( 8 * 8, 8 * 8 * 8)
        self.fc3e = nn.Linear(8 * 8 * 8, 16 * 16 * 16)
       
        self.conv_logvar2 = nn.Conv2d(16, 1, 1, stride = 1, padding=0)
        self.conv_mu_z1 = nn.Conv2d(16, 1, 1, stride = 1, padding=0)
 
        self.upconv0 = nn.ConvTranspose2d(1, 16, 1, stride=1, padding=0)
 
        self.upconv1 = nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1)
         
        self.upconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
          
        self.upconv3 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)
           

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2, stride=2)  
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.to(device)


    def step(self, z1):
      
        return self.Visual_Dec(z1), self.Prop_Dec(z1)


    def prediction(self, x, Im):
        z1_v, logvar_v = self.Visual_Enc(Im)
        z1_p, logvar_p = self.Prop_Enc(x)
        return self.Visual_Dec(z1_v + z1_p), self.Prop_Dec(z1_v + z1_p), (z1_v + z1_p)
        

    def forward(self, x, y):
        
        z_visual, z_visual_logvar = self.Visual_Enc(y)
        z_prop, z_prop_logvar = self.Prop_Enc(x)
        z1 =  self.reparameterize(z_visual, z_prop, z_visual_logvar)
        output_y = self.Visual_Dec(z1)
        output_x = self.Prop_Dec(z1)
        return output_y, output_x
        
    def perception(self, z1, Im_std, Im_attr, mu_std, mu_attr):

       z1 = Variable(z1, requires_grad=True)
   
       # Prediction of Image and joints 
       Out_im, Out_mu = self.step(z1)

       # Initialization grad in the trees graph of the network  
       z1.grad = torch.zeros(z1.size(), device=device, dtype=torch.float, requires_grad=True)
       Out_mu.backward(mu_std*(mu_attr - Out_mu)/0.001, retain_graph=True)
       grad_mu = torch.clone(z1.grad)
       # Backward pass for both Image and joints 
       z1.grad = torch.zeros(z1.size(), device=device, dtype=torch.float, requires_grad=True)
       Out_im.backward(Im_std*(Im_attr - Out_im)/0.0135, retain_graph=True)# 0.005
       grad_Im = torch.clone(z1.grad)

       return Out_im, Out_mu, grad_Im, grad_mu      
 

    def reparameterize(self, mu_v, mu_p, logvar):
  
        eps = torch.randn_like(logvar)
        return eps * torch.exp(0.5 * logvar) + mu_p + mu_v

    def Visual_Enc(self, x):  #Visual Encoder
        
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.avgpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.relu(self.upconv3e(x))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = self.relu(self.upconv4e(x))
        logvar = self.maxpool(self.relu(self.conv_logvar(x)))
        x = self.maxpool(self.relu(self.conv_Im_z1(x)))
        return x, logvar
        

    def Visual_Dec(self, x):   #Visual Decoder
        x = self.relu(self.upconv0(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.tanh(self.relu(self.upconv3(x)))
        
        return x


    def Prop_Enc(self, x):  #Proprioceptive Encoder
        
        x = self.relu(self.fc1e(x))
        x = self.relu(self.fc2e(x))
        x = self.relu(self.fc3e(x))
        x = x.view(-1, 16, 16, 16)
        logvar = self.relu(self.conv_logvar2(x))
        x = self.relu(self.conv_mu_z1(x))
        return x, logvar
        

    def Prop_Dec(self, x):   #Proprioception Decoder
        
        x = x.view(-1, 16 * 16)
        x = self.relu(self.fc1d(x))
        x = self.relu(self.fc2d(x))
        x = self.relu(self.fc3d(x))
        return x

    def Prop_Dec_action(self, x):   #Proprioception Decoder
        
        x = x.view(-1, 16 * 16)
        x = self.relu(self.fc1d(x))
        x = self.relu(self.fc2d(x))
        x = self.relu(self.fc3d(x))
        return x[0]

    def get_dictionaries(self):

       partition = { 'train' : {}, 'validation' : {} }
       labels = {}
   
       train_list = []
       validation_list = []
       labels_list = []

       for i in range(650):
           train_list.append(str(i))

       partition['train']= train_list

       for i in range(200):
           validation_list.append(str(i+650))

       partition['validation']= validation_list

       for i in range(850):
           labels[str(i)] = str(i)

       return partition, labels
           
       

    def train_net(self, net, network_id, max_epochs , batch_size):
       

        torch.cuda.empty_cache()
        net.to(device)

        partition, labels = self.get_dictionaries()

        params_train = {'batch_size': 64,
                  'shuffle': True,
                  'num_workers': 4}

        params_val = {'batch_size': 1,
                  'shuffle': True,
                  'num_workers': 4}
        

        # Generators
        training_set = Dataset(partition['train'], labels)
        training_generator = DataLoader(training_set, **params_train)

        validation_set = Dataset(partition['validation'], labels)
        self.validation_generator = DataLoader(validation_set, **params_val)
    
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)

        criterion = nn.MSELoss()

        epoch_loss = []
        val_loss = []
        batch_loss = []
        fig, ax = plt.subplots(3, 3, figsize=(16, 12))
        plt.ion()


        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []


            net.train()
            for local_batch, local_labels in training_generator:

                loss, MSE_Im, MSE_z1, output_x = AutoEncoder.run_batch(local_batch, local_labels, True, net, optimizer, criterion, device)
                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 and epoch<max_epochs-1:

                for local_batch, local_labels in self.validation_generator:
                    net.eval()
                    loss, MSE_Im, MSE_mu, output_x = AutoEncoder.run_batch(local_batch, local_labels, False, net, optimizer, criterion, device)
                    cur_val_loss = np.append(cur_val_loss, [loss])


                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])
                self.get_error(net, epoch, ax)
                print('------ Epoch ', epoch, '--------LR:', scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])

                torch.save(net.state_dict(), AutoEncoder.SAVE_PATH + "/" + network_id + "/trained_network" + network_id)  

                ax[0,0].set_title("Loss")
                ax[0,0].plot(range(len(epoch_loss)), epoch_loss, label="Epoch loss")
                ax[0,0].plot(np.arange(len(val_loss))*10, val_loss, label="Validation loss")
                #plt.pause(0.001)
                
        torch.save(net.state_dict(), AutoEncoder.SAVE_PATH + "/" + network_id + "/trained_network" + network_id)


    def run_batch(x, y, train, net, optimizer, criterion , device):
        """
        Execute a training batch
        """
      
        q = torch.tensor(np.float32(x), device=device)
        input_y = torch.tensor(np.float32(y), dtype=torch.float, device=device)
        target_y = torch.tensor(np.float32(y), dtype=torch.float, device=device, requires_grad=False)
       
       
        optimizer.zero_grad()
        output_y, output_x = net(q, input_y)

        MSE_Im = criterion(output_y, target_y)
        MSE_mu = criterion(q, output_x)
        loss = MSE_Im + MSE_mu

        if train:
            loss.backward()
            optimizer.step()

        return loss.item(), MSE_Im, MSE_mu, output_x

    def load_from_file(self, model_id):
        """
        Load network from file
        :param model_id: save id to load from
        """
        self.load_state_dict(torch.load(os.path.join(self.SAVE_PATH, model_id + "/trained_network" + model_id), map_location='cuda:0'))
        self.eval()
  



