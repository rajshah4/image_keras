
##DRAFT - PLEASE DO NOT USE YET

### Building an image classifier using keras

The fact that computers can see is just not that amazing anymore.  But, the techniques for teaching a computer to do this are now simpler and more refined then ever.  This post shows you how easy it is to build an image classifier.  All you need is some basic python skills and at least a few hundred images.  
  
If you work your way through the blog post, you will be able to put together a customized image classifier that will be close to the state of art (and blow away models from a few years ago).  
  
The approach here uses Keras, which is emerging as the best library for building neural networks. The code here also assumes you are using Tensorflow as the underlying library (it won’t run properly if you are using Theano).  
  
### To run the code you need to:  
- Get a keras docker container running  
- Download the data  
- Work your way through the notebook  

Let me step you through each of these steps.  

#### Get a keras docker container running  
Much of deep learning is built around python.  Python has many libraries that have inner dependencies that make it difficult to replicate work.  Deep learning teams now rely on technologies like docker that make it easy to abstract away these dependencies.  It might be a pain to learn, but if you are doing deep learning without docker, you are building an island.
I have tested the notebook with a docker container `ermaker/keras-jupyter`.  This docker container includes keras and tensorflow.  After you have installed docker on your system, you the follow command should get you up and running:  

`docker run -d -p 8888:8888 -e KERAS_BACKEND=tensorflow -v /Users/rajivshah/Code:/notebook ermaker/keras-jupyter`  

You will just want to substitute your path to the notebook and data for `/Users/rajivshah/Code`  

#### Download the data  
There are two major sources of data needed to run this notebook:  
- Kaggle data of the images of [cats and dogs](https://www.kaggle.com/c/dogs-vs-cats/data)  
- [VGG16 weights](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3*) for a pretrained network 
The images are already in the github repo, however, it will be necessary to download the VGG weights and place them in the model/vgg directory.  

#### Work your way through the notebook
The notebook is based off a brilliant blog post by Francois Chollet @fchollet https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html and a workbook by Guillaume Dominici https://github.com/gggdominici/keras-workshop  

Go ahead and read the full blog post to learn all the details.  But generally, the approach is to start with a small model and then improve it by augmenting it with more data.  The notebook then discusses using a pretrained network VGG16 and ends by showing how to fine tune a pretrained network by modifying the last layers.  
  
The problem with the blog post is that the code listed doesn’t work when you have tensorflow as a back end.  If you read the comments to the code gists, you will find a lot of frustrated people.  I spent a lot of time and picked up hints from a lot of places when putting this notebook together.  
