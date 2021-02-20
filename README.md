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




TEST OUTPUTS
==================================================
                               ACCURACY = 0.8053
==================================================
**************************************************
Current Learning Rate:  0.001
Batches:  10
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6039120679942847
FCE for Testing Set:  0.6260058672435643
**************************************************
==================================================
                               ACCURACY = 0.8069
==================================================
**************************************************
Current Learning Rate:  0.005
Batches:  10
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6063566200143571
FCE for Testing Set:  0.6290254878615034
**************************************************
==================================================
                               ACCURACY = 0.8061
==================================================
**************************************************
Current Learning Rate:  0.01
Batches:  10
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6063601375515464
FCE for Testing Set:  0.6286650948449042
**************************************************
==================================================
                               ACCURACY = 0.8053
==================================================
**************************************************
Current Learning Rate:  0.005
Batches:  50
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6036250037194342
FCE for Testing Set:  0.6261157219369525
**************************************************
==================================================
                               ACCURACY = 0.8056
==================================================
**************************************************
Current Learning Rate:  0.01
Batches:  50
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6039935105810144
FCE for Testing Set:  0.626657709846517
**************************************************
==================================================
                               ACCURACY = 0.7992
==================================================
**************************************************
Current Learning Rate:  0.005
Batches:  100
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.607187213028079
FCE for Testing Set:  0.626284573251762
**************************************************
==================================================
                               ACCURACY = 0.8047
==================================================
**************************************************
Current Learning Rate:  0.01
Batches:  100
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.6035178592868317
FCE for Testing Set:  0.6262783920799203
**************************************************
==================================================
                               ACCURACY = 0.7982
==================================================
**************************************************
Current Learning Rate:  0.01
Batches:  200
Epochs 50
L2-Reg 0.02
FCE for Training Set:  0.60965658528204
FCE for Testing Set:  0.6329161171390287
**************************************************
