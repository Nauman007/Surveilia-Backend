import matplotlib.pyplot as plt
import numpy as np

Train_loss_values = np.load('Myplots/TrainingLoss.npy')
val_lost_values = np.load('Myplots/val_lost_values.npy')                   

                 
TrainAccuracy = np.load('Myplots/TrainAccuracy.npy')
ValidationAccuracy =np.load('Myplots/ValidationAccuracy.npy')                 

plt.tight_layout()

plot1 = plt.figure(1)
plt.plot(Train_loss_values,label = 'Training Loss')
plt.plot(val_lost_values, label = 'Validation Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

plot2 = plt.figure(2)
plt.plot(TrainAccuracy,label = 'Training Accuracy')
plt.plot(ValidationAccuracy, label = 'Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(loc = 'lower right')
plt.show()
