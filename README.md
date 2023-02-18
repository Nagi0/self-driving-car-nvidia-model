# self-driving-car-nvidia-model
Making a self driving car using Nvidia model on a simulator

# WARNING: This code uses Tensorflow-GPU, make sure to install Nvidia Toolkit 11.0-11.2. Currently those are the only versions compatible with Tensorflow-GPU. Make sure you have the CuDNN that corresponds to your Toolkit version. If you do not have an Nvidia graphic card you can remove Tensorflow-GPU from the requirements.txt and install Tensorflow, it will use your CPU insted. <br>
## Step 1:
Download the driving simulator "Term1" from the following repository: <br>
https://github.com/udacity/self-driving-car-sim/blob/master/README.md

## Step 2:
Create an empity folder and call it "myData"

## Step 3:
Record your data in the training mode and choose the myData folder to save the data <br>
  ### Warning: It is recommended that you use your keyboard to record data, do from 3 to 5 laps; stop recording; turn the car arround; and do from 3 to 5 additional laps.
  
## Step 4:
Run the main code, check out the histogram that will be plotted, there is a part of the code where we balance our data, this is why two histograms are plotted. It indicates the ammount of data for each Steering Angle values. Usually there is a peak of values in the zero. The graph below shows the not balanced data:
![alt text](https://raw.githubusercontent.com/Nagi0/self-driving-car-nvidia-model/main/graph_1.png)
The following is the balanced data graph: 
![alt text](https://raw.githubusercontent.com/Nagi0/self-driving-car-nvidia-model/main/graph_2.png)
Now, if your graph above istill has too much zero values, go to the "utils.py" code, there you can change the variable "samples_bin" to suit better to the data you have. Just remember that we want to have a good ammount of zero values, because most of the time the car goes straight, due the fact we used the keyboard to record data.

## Step 5:
Wait it to finish the training and run the "test_model.py" code. Now open the simulator again and select the autonomus option, wait a little to the code connect, and see the results <br>
### Note: You can change the training option on the "history" variable, changing the number of epochs, steps_per_epoch or validation_steps, can make your model more or less accurate, changing those values will affect the traning time. 
