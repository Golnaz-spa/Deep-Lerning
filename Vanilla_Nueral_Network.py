import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist

"""
Object dDetection: we have number of picture . We want to use deep learning method to detect the picture
Deep learning has 2 levels: 1- define layers  2- compile the Network
"""
l = fashion_mnist.load_data()
print(l)
print(type(l))
print(len(l))
print("x_train" ,l[0][0].shape)
print("y_train", l[0][1].shape)

#Read the training data using loadlocal_mnist function-(X,y) is train part  that is first part
((X,y), (Xtest, ytest)) = fashion_mnist.load_data()

#we have 10 classes of pictures. Insteda of name of picture used numbers
print(np.unique(y)) // array(0,1,2,3,4,5,6,7,8,9)

#change the pixel intensities to continuous variables
X=X/255
print("shape X: ",X.shape)
print("First Picture", X[0,:,:])
plt.imshow(X[0,:,:],cmap='gray')


##Define the model. Sequential is the easiest way to build a model in Keras.
##It allows you to build a model layer by layer
##Layer 1: a dense (fully-connected) layer with 100 hidden units. The activation function is ReLU.
##Layer 2: the last fully connected layer with 10 units (because we have 10 classes)
##You can use more than one fully-connected layers with different number of neurons

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

##Compile the model
##Choose the optimizer. The ADAM optimizer is selected. It is an efficient version of the stochastic gradient descent
##Choose the loss function. Cross-entropy is used since we have multiple calsses.

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#we should scale the test data
Xtest=Xtest/255

#we can fit the model. here fit means define weight and bias
model.fit(X,y,batch_size=50,epochs=10,verbose=2,validation_split=0.1)

#Compute the fitted values (probabilities of classes)
yhat=model.predict(Xtest)
print(yhat.shape)
"""
print(yhat.shape) --> it gives (10000,10). it means the pobability of first one is 2.01342e-10, and so on. for other 10 classes. 
yhat has 10 column due to we have 10 class(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""

#probabilites - for finding accuracy of the model- verbose=2 we put 2 to give the whole aacuracy if we put 1 give accuracy for each epoch
model.evaluate(Xtest,ytest,verbose=2)


#Now we want to "Save our model" for future use for prediction
model.save('.../vanilla_model.h5')
#in ... put the address of the file than we want save the model from that folder. If we do not put the path it save in directory
#be careful we use .h5 format for saving the model. vanilla_model is the name of the model

load_model=tf.keras.models.load_model('.../vanilla_model.h5')
#we load the model here
load_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# we want to use new data to find the result of the model
im=plt.imread('...image3.jpg')
#we see the image 3
plt.imshow(im)

#shape is (28*28*3)- it is a colorful picture should convert it to gray one as the model work with gray picture
print(im.shape())

# convert to gray picture
im=np.mean(im,2)
plt.imshow(im,cmap='gray' )
print(im.shape)
#(28*28)

#background of picture 'image3' is different from fashion_mnist picture that used to train the model. so need to change it
im1=255-im

#scale the data
im1=im1/255

#use reshape to convert to picture 28*28 to a vector (1*784)
im1=np.reshape(im1,(1,784))

#predict the label of the class
predict=load_model.predict(im1)

#we have many pictures of these items
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names[np.argmax(predict)]











