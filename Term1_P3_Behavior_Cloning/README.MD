This is the 3rd project in Term 1, Behavior Cloning. 

The project is based on the drving simulator provided by the course. Using the simulator, we can collect a lot of good driving
behavior data including the steering angle and the image. By feeding these images and steering angles to a convolutional neural
network built in Keras, we can train a model to predict steering angle based on image input and thus use the model to control
the vehicle to drive itself in the simulator. 

The details of this project can be found in [writeup](writeup.pdf). 
1. model.py contains the final convolutional neural network model used in this project
2. drive.py is the code to communicate with the simulator, including get image from the simulator and send back the predicted 
steering angle value.
3. model.h5 contains the model trained.

The result of this project can be found [here](https://www.youtube.com/watch?v=UaAnEgwRS_o&t=24s)
