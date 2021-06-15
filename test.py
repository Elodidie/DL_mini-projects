"""
This file is the main file that executes the models and print the results.
"""
""""""

import torch 
from torch import nn
from torch.nn import functional as F
import torch.optim as optim


from tools import * 

from Model_Architectures.CNN import * 
from Model_Architectures.CNN_with_auxiliary import *
from Model_Architectures.MLP import *
from Model_Architectures.MLP_with_auxiliary import *

import dlc_practical_prologue as prologue


def train_model(model, x_tr, y_tr, classes_tr, x_test, y_test, mini_batch_size, epochs, aux, aux_hyper ) :
    loss_function = nn.BCELoss()
    auxiliary_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr =  0.0001)
    
    Train_loss = torch.zeros(epochs)
    Accuracies = torch.zeros(epochs)
    
    for e in range(epochs):
        model.train()
        acc_loss = 0

        for b in range(0, x_tr.size(0), mini_batch_size):
            
            #CNN or MLP
            if aux == False : 
                output = model(x_tr.narrow(0, b, mini_batch_size))
                loss = loss_function(output, y_tr.narrow(0, b, mini_batch_size).float())
            #Siamese CNN or Siamese MLP
            else : 
                output1, output2, output3 = model(x_tr.narrow(0, b, mini_batch_size))
            
                loss1 = loss_function(output1, y_tr.narrow(0, b, mini_batch_size).float())
                loss2 = auxiliary_loss_function(output2,classes_tr[:,0].narrow(0, b, mini_batch_size))
                loss3 = auxiliary_loss_function(output3,classes_tr[:,1].narrow(0, b, mini_batch_size))
                loss = loss1 + aux_hyper*(loss2 + loss3) 
            # total loss for each batch
            acc_loss = acc_loss + loss.item() * mini_batch_size 
            
            #Optimization step 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        #Train loss for each epoch
        Train_loss[e] = acc_loss / x_tr.size(0)
        #Test accuracy
        Accuracies[e],_,_ = compute_metrics(model, x_test, y_test, aux )
      
    return Train_loss , Accuracies



def main() : 
    torch.manual_seed(1234)
    #number of data in train and test
    nb = 1000
    mini_batch_size = 10
    nb_epochs = 30
    
    #Models Architectures to see all the results
    archis_params = [('CNN', False, None) , ('CNN_aux', True, 1.5), 
                    ('MLP', False, None),  ('MLP_aux', True, 1)]
    
    archis_params = [('CNN_aux', True, 1.5)]
    # Number of different runs for each model
    boucle =  2
    # Only show one model for one run 

 
    # plot the figures or not
    Figures = False
    
   #Empty arrays to store the metrics + standard deviations
    models_acc=[];models_acc_std=[];models_rec=[];models_spe=[];models_params=[] 
    models_rec_std = [] ; models_spe_std = []
    
     
    
    for archi in archis_params : 
        ACC = torch.zeros(boucle)
        SPE = torch.zeros(boucle)
        REC = torch.zeros(boucle)
    
        for i in range(boucle) :
        # initialize the model
            
            # random data : 
            train_input,train_target,train_classes,test_input,test_target,test_classes = prologue.generate_pair_sets(nb)
            # initialize the model
            model_name = archi[0]
            model = globals()[model_name]()
            model.train()
            
            # Model parameters
            auxiliary = archi[1]
            hyper = archi[2]
         
            # train the model 
            epoch_loss, epoch_acc = train_model(model,train_input,train_target,train_classes, test_input, test_target , mini_batch_size,nb_epochs, auxiliary,hyper)
            
            # only plot the train loss and test accuracy vs epochs for one run
            if i == 1 and Figures == True : 
                plot(epoch_loss, epoch_acc, model_name)
            
            # Compute Accuracy, Recall and Precision
            model.eval()
            ACC[i],REC[i],SPE[i] = compute_metrics(model, test_input, test_target, auxiliary)
            
        
        #Store the results of the metrics.
        models_acc.append(ACC.mean().item()) ; models_rec.append(REC.mean().item()) ; 
        models_spe.append(SPE.mean().item()) ;
        models_acc_std.append(ACC.std().item())
        models_rec_std.append(REC.std().item())
        models_spe_std.append(SPE.std().item())
        models_params.append(model.number_parameters())
    
    """Print all the results""" 
    if len(archis_params) == 4 : 
        print_results(['CNN', 'CNN_aux', 'MLP','MLP_aux'] , models_acc, models_acc_std, models_rec, models_rec_std, models_spe, models_spe_std, models_params)
    elif len(archis_params) == 1 : 
        print_results(['CNN_aux'] , models_acc, models_acc_std, models_rec, models_rec_std, models_spe, models_spe_std, models_params)
    
main()