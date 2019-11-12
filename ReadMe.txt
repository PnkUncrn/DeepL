**This file contains a short  description of all the files in the DeepL directory.
This is a working copy**

1-MLP_MNIST_arch.lua
	Code generates MLP architecture used for MNIST in Torch 7. ITs written in LuaJIT and is WORKING code.
	Only generated the NN. No training or testing. Can be excuted.
	Refer figure 5  of paper "Privacy Preserving Deep Learning"

2-MLP_MNIST_Arch2.lua
	Code generates MLP architecture for MNIST. Includes creation of the NN, training and testing.
	Code is currently non functional. 

3-dataset_MNIST.lua
	Helper file for MLP_MNIST_Arch2.lua. 
	Normalizes the MNIST dataset before it is used to train the NN in MLP_MNIST_Arch2.lua
	Source : https://github.com/torch/demos/blob/master/train-a-digit-classifier/dataset-mnist.lua
	Did not write this code.

cod.lua
	Simple test file.
	Unrelated to project.

