import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 

class Neural(nn.Module):
    def __init__(self, layer_size, active_fun, batch):                 ## initialize a neural. Input the activation function
        super().__init__()                                                        ## layer_size is the neuron num of each layers
        self.active_fun=active_fun
        self.layer_size=layer_size
        self.batch=batch
        self.error=np.array([])            ## record the error 
        self.weight=nn.ParameterList([nn.Parameter(0.1*torch.rand(layer_size[i]+1,layer_size[i+1],requires_grad=True)) for i in range(len(layer_size)-1)])  ## initialize the weight of each layer
        

        #print("weight", *self.weight)
           
            
            

                
                
                
    def forward(self,  X   ):                                                   ## update the value in the neural
        
        
        Predict=[]  ## initialize the prediction
        for x in X:      
            
            Front_layer_val = x  
                                            
            for layer in range(len(self.layer_size)-1 ):
                
                bias=torch.tensor([1],dtype=torch.float32,requires_grad=True)                                                ## add bias to the layer
                Front_layer_val=torch.cat([Front_layer_val,bias],dim=0)                              ## Add bias to each layer

                Front_layer_val   = torch.matmul(Front_layer_val, self.weight[layer])
                active_fun=self.active_fun[layer]
                
                if active_fun=='sigmoid':
                    Front_layer_val = 1 / (1 + torch.exp(-Front_layer_val))                                  ## activation function sigmoid
                elif active_fun=='relu':
                    Front_layer_val = torch.max(Front_layer_val, torch.tensor([0.0]))                           ## activation function relu
                elif active_fun=='tanh':
                    Front_layer_val = torch.tanh(Front_layer_val)                                              ## activation function tanh
                elif active_fun=='linear':
                    Front_layer_val = Front_layer_val

            Predict.append(Front_layer_val)                                                                 ## append the prediction

        return Predict
        
    
     
            
            
            

    def loss(self,X, actual_result):                                                                 ## get the loss function
        
        
        prediction = self.forward(X)                                                                 ## get the prediction)
        loss=torch.tensor(0.0,requires_grad=True)
        
        for pred,act_result in zip(prediction,actual_result):
            single_loss=torch.sum(torch.pow((pred - act_result ), 2)) / self.layer_size[-1]
            if loss==0:
                loss=single_loss
            else:
                loss=loss+single_loss
            
        #print("x",X)
        #print("pred",prediction) 
        #print("act",actual_result)
        #print("loss",loss)
        loss=loss/self.batch
   
        self.error=np.append(self.error,[loss.detach().numpy()])                                                                  
        return loss          ## loss function    
            
        
    

    def get_grad(self,X, actual_result):                                                                 ## get the gradient of the loss function with respect to the weight
        
        loss_fun=self.loss(X, actual_result)                                                                 ## get the loss function
        
        loss_fun.backward()                                                                                     ## backward the loss function
        
        
        



        grad_loss=[weight.grad for weight in self.weight]                                                     ## get the gradient of the loss function with respect to the weight
        
               
        return grad_loss                                                                   ## return the gradient of the loss function with respect to the weight
        
    def train(self, train_X, train_Y, learning_rate ):                                                                 ## train the neural
        batch=self.batch
        
        for i in range(0,len(train_X),batch):
            
            
            X_batch=[train_X[j] for j in range(i,i+batch)]## iterate the training data
            
            
            Y_batch=[train_Y[j] for j in range(i,i+batch)]  ## iterate the training data
            
            grad=self.get_grad(X_batch,Y_batch)                                                             ## get the gradient of the loss function with respect to the weight
               
            
            with torch.no_grad():                                                                     ## iterate the weight
                self.weight =nn.ParameterList([nn.Parameter(old_weight - learning_rate * grad)
                                            for old_weight , grad in zip(self.weight,grad)])        ## update the weight
            
                                                                                                       ## return the weight
        
        return self.weight
    def model(self, test_X):                                                                 ## test the neural
        return self.forward(test_X)                                                                 ## return the prediction

'''
sample=Neural([1,10,1],'relu',10)
num=3000
X = [(torch.rand(1) ).clone().detach().to(torch.float32) for i in range(num)]               ##idk why clone and detach is needed    
Y   = [x**2 for x in X]
for epoch in range(20):
    sample.train(X,Y,0.005)
print(X[15])
output=sample.model([X[15]])

print(output)

'''
sample=Neural([784,6,3,1],['tanh','relu','linear'],30)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = nn.ParameterList()
Y_train = nn.ParameterList()
X_test = nn.ParameterList()
Y_test = nn.ParameterList()


for i, data in enumerate(train_images):
    X_train.append(torch.tensor(data.flatten()).clone().detach().to(torch.float32) )
    y=torch.zeros(10)
    y[train_labels[i]]=1
    Y_train.append(y)
X_train_tensor = torch.stack(list(X_train), dim=0)
mean = X_train_tensor.mean(dim=0, keepdim=True)  
std = X_train_tensor.std(dim=0, keepdim=True)    
epsilon = 1e-8
std = std + epsilon
X_train_norm = (X_train_tensor - mean) / std  # shape: (N, 784)


for i, data in enumerate(test_images):
    
    X_test.append(torch.tensor(data.flatten()).clone().detach().to(torch.float32) )
    y=torch.zeros(10)
    y[test_labels[i]]=1
    Y_test.append(y)
    
sample.train(X_train,train_labels,0.01)
print(sample.model([X_train[8]]))
print(test_labels[8])




plt.plot(sample.error)
plt.show()
print(sample.error[-1])

