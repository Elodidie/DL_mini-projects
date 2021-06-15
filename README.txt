# EPFL Course - Deep Learning - 

## Content 
***
We study 4 different NN architectures to find the order of magnitude on 2 didits, and compare the performance
of each model.

##Running the program 
***
The main file is test.py, it requires no arguments. It executes only one model with 30 epochs and 2
different runs. It prints the results in the terminal and runs for around 8min on a 
standard computer, including the loading of the data.

To have all the results, run the 4 models with 10 runs for each model. In order to do that, one needs
to put line 77 of test.py as comments in order to take the previous values and put the boucle variable
to 10. For all the results, it runs for around 1 hour. 


To run the code : python test.py

##Files : 
***
Cross_validation.py : performs cross validation to choose best hyperparameters
dlc_practical_prologue : loads the data
test.py : main 
tools.py : include useful functions 
Model_Architectures : contains a python file for each model architecture


##Librairies :
***
Python 3.7
PyTorch
Pandas
Matplotlib

#Authors: 
***
Alexandre Di Piazza
Elodie Adam
Clémentine Lévy-Fidel