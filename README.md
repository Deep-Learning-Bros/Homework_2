# Homework_2

We used the following values to optimize the hyperparameters using a grid search. As per instruction, we used 4 sets of values for each hyperparameter to tune the model. We also included some of our outputs by the model for accuracies above 80% in this report. Our best result was 80.69% accuracy and FCE of 0.62. In addition, to tuning hyperparameters we also, normalized the data by (1/255). This led to better overall predictions.
Learning Rate = [0.0001, 0.001, 0.005, 0.01]
Mini-Batch Sizes = [10, 50, 100, 200] 
L2 Regularization Strength = [0.02, 0.05, 0.07, 0.1] 
Number of Epochs = [5, 10, 50, 100] 
Best Results Hyperparameters 
Learning Rate = 0.005 
Batches = 10 
Epochs = 50 
L-2 Regularization Strength = 0.02 
FCE for Training Set:  0.6063566200143571
FCE for Testing Set:  0.6290254878615034



