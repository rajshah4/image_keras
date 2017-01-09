
### Building an image classifier using keras

The fact that computers can see is just not that amazing anymore.  But, the techniques for teaching a computer to do this are now simpler and more refined than ever.  This post shows you how easy it is to build an image classifier.  All you need is some basic python skills and at least a few hundred images.  
  
The approach here uses Keras, which is emerging as the best library for building neural networks. The code here also assumes you are using Tensorflow as the underlying library (it won’t run properly if you are using Theano).  
  
### To run the code you need to:  
- Get a keras docker container running  
- Download the data  
- Work your way through the notebook  

Lets step through each of these steps

#### Get a keras docker container running  
Much of deep learning is built around python. Python has many libraries that have conflicting dependencies that make it difficult to replicate work. Deep learning teams now rely on technologies like docker that abstract away these dependencies.  It might be a pain to learn, but if you are doing deep learning without docker, you are building an island.  
  
I have tested the notebook on an Ubuntu OS with a docker container `ermaker/keras-jupyter`. Another alternative docker container is available from [gwlabs](http://gw.tnode.com/docker/keras-full/).  This docker container includes keras and tensorflow.  After you have installed docker on your system, the following command should get you up and running:  

`docker run -d -p 8888:8888 -e KERAS_BACKEND=tensorflow -v /Users/rajivshah/Code:/notebook ermaker/keras-jupyter`  

You should substitute your local path to the notebook and data for `/Users/rajivshah/Code`  

You can skip using a docker container, but make sure to have tensorflow and keras installed on your machine.

#### Download the data  
There are two major sources of data for this notebook:  
- Kaggle data of the images of [cats and dogs](https://www.kaggle.com/c/dogs-vs-cats/data)  
- [VGG16 weights](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) for a pretrained network 
A sample of Kaggle images of cats and dogs are already in the github repo, however, it is necessary to download the VGG weights and place them in the model/vgg directory.  

#### Work your way through the notebook
The notebook is based off a brilliant blog post by Francois Chollet @fchollet https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html and a workbook by Guillaume Dominici @gggdomi https://github.com/gggdominici/keras-workshop  

You should read the full blog post to understand the purpose of the notebook and the underlying details.  In sum, the blog post covers a set of techniques for building an image classification model using a small amount of image data.  It starts with building a small convolutional neural network and then improving it by augmenting the model with more data.  The notebook then discusses using a pretrained network VGG 16 layer model and ends by showing how to fine tune a pretrained network by modifying the last layers.  
  
The problem with the blog post is that the code listed doesn’t work when you have tensorflow as a back end.  If you read the comments to the code gists, you will find a lot of frustrated people.  I spent a lot of time and picked up hints from a lot of places when putting this notebook together. The results should be a smooth experience that can let you focus on building an image classifier.

The first notebook only contains code that is relevant to running models according to Chollet's blog post.  I also created a second notebook (notebook_extras) that contains additional code for those who want to inspect the training data, validate their predictions, get more details on the models, use tensorboard, or use [quiver](https://github.com/jakebian/quiver) to visualize the activation layers in their models.
