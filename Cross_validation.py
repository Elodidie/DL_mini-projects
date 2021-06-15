""" 
This files uses Cross Validation to find the best hyper__parameter for the auxiliary
function in the models using an auxiliary function
"""

import torch 
from tools import * 
from Model_Architectures.CNN_with_auxiliary import *
from Model_Architectures.MLP_with_auxiliary import *
import numpy as np 
import dlc_practical_prologue as prologue

torch.manual_seed(1234)

def train(model, x_tr, y_tr, classes_tr, mini_batch_size, epochs,aux_hyper ) :
    loss_function = nn.BCELoss()
    auxiliary_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 5*0.0001)
    
    for e in range(epochs):
        model.train()

        for b in range(0, x_tr.size(0), mini_batch_size):
            
            optimizer.zero_grad()
            output1, output2, output3 = model(x_tr.narrow(0, b, mini_batch_size))
            
            loss1 = loss_function(output1, y_tr.narrow(0, b, mini_batch_size).float())
            loss2 = auxiliary_loss_function(output2,classes_tr[:,0].narrow(0, b, mini_batch_size))
            loss3 = auxiliary_loss_function(output3,classes_tr[:,1].narrow(0, b, mini_batch_size))
            loss = loss1 + aux_hyper*(loss2 + loss3) 
     
            loss.backward()
            optimizer.step()
        
def indices(x_train, y_train, y_class, k_fold, index ):
    "returns the indices for index i for the cross validation"
    n = x_train.size()[0]
    interval = int(n/k_fold)
    rand_indices = torch.randperm(n)
    k_indices = [rand_indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    test_indices = k_indices[index]
    train_indices_list = [val for is_good,val in zip(~(torch.arange(k_fold) == index),k_indices) if is_good]
    train_indices = [val for sublist in train_indices_list for val in sublist]
    
    x_tr = x_train[train_indices] ; y_tr = y_train[train_indices] ; class_tr = y_class[train_indices]
    x_val = x_train[test_indices] ; y_val = y_train[test_indices] ; class_val =   y_class[test_indices]
    return x_tr, y_tr, class_tr, x_val, y_val, class_val 


def cross_val(model_name, parameters, k_fold, mini_batch_size, epochs) :
    "Performs the Corss Validation on k_fold"
    results = np.zeros(k_fold)
    index = 0 
    nb = 1000
    train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(nb)
    for hyper in parameters : 
        Accuracies = []
        for k in range(k_fold) : 
            # initialize the model
            model = globals()[model_name]()
            # take indices for train and validation set
            x_tr, y_tr, class_tr, x_val, y_val, class_val  = indices(train_input,train_target,train_classes, k_fold, k)
            train(model, x_tr, y_tr, class_tr, mini_batch_size, epochs, hyper )
            # Compute accuracy of the model
            Acc,_,_ = compute_metrics(model, test_input, test_target, True )
            Accuracies.append(Acc)
        
        results[index] = sum(Accuracies)/len(Accuracies)
        index +=1 
    # Take the parameter corresponding to the best accuracy
    best = np.argmax(results)
    return parameters[best]

# Try 5 different parameters    
hypers = [0, 0.5, 1, 1.5, 2]
batch_size = 10
epochs = 25
kfold = 5
hyper1 = cross_val('CNN_aux', hypers, kfold, batch_size, epochs)
hyper2 = cross_val('MLP_aux', hypers, kfold, batch_size, epochs)
print(hyper1, hyper2)

