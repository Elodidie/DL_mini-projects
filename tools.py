"""
This file contains the functions used in the main file
"""
import torch
import numpy as np 
import pandas as pd


#This function measures the predictions of the model
def compute_metrics(model, x, y, aux) : 
    TP = 0 
    FP = 0 
    TN = 0 
    FN = 0 
    for j in range(x.size()[0]) :
        if aux == False : 
            output = model(x.narrow(0,j, 1))
            
        else : 
            output, a, b = model(x.narrow(0,j, 1))
        
        prediction = 1*(output>0.5)    
      
        real = y[j].item()

        if prediction == 1 : 
            if real == 1: 
                TP +=1
            else : 
                FP +=1
        else : 
            if real == 0 :
                TN +=1
            else : 
                FN +=1
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall=TP/(TP+FN) if TP + FN != 0 else 0
    specificity = TN/(TN+FP) if TP + FP != 0 else 0 
    
    return accuracy, recall, specificity

#Plot the train_loss and test_accuracy vs epochs
def plot(train_loss, accuracy, model_name) : 
    import matplotlib.pyplot as plt 
    fig, (ax1, ax2) = plt.subplots(1,2 , sharex = True )    
    ax1.plot(train_loss)
    ax2.plot(accuracy)
    
    ax1.set(xlabel ='epochs', ylabel = 'Train Loss')
    ax2.set(xlabel ='epochs', ylabel = 'Test Accuracy')
    fig.tight_layout(pad=3.0)
    fig.suptitle('Loss and Accuracy for Model : ' + model_name)
    plt.savefig('Figures/' + model_name + '_results')

#Print all the results
def print_results(models, acc, acc_std ,recall, recall_std, specificity,specificity_std,
                  parameters):
    data = {'Models' : models , 'Accuracies' : acc, 'Accuracies std' : acc_std, 
            'Specificity' : specificity, 'Specificity std' : specificity_std ,
            'Recall' : recall, 'recall' : recall_std, 'Nbr_parametres' : parameters}
    df = pd.DataFrame(data)
    #df.to_csv('Figures/results')
    print(df)
    
