**This file contains a short  description of all the files in the DeepL directory.
This is a working copy**

1-Final.lua
	Code generates MLP architecture used for MNIST in Torch 7. ITs written in LuaJIT and is WORKING code.
	Generation and Testing of NN and premise of the research. 
	Refer figure 5  of paper "Privacy Preserving Deep Learning"
	No longer in use.

3-dataset_MNIST.lua
	Helper file for MLP_MNIST_Arch2.lua. 
	Normalizes the MNIST dataset before it is used to train the NN in MLP_MNIST_Arch2.lua
	Source : https://github.com/torch/demos/blob/master/train-a-digit-classifier/dataset-mnist.lua. edited

4-cod.lua	
	Testing file. Updated Regularly with new testing scenarios.
	No longer in Use
	
5-NeuralNets.lua
	Contains Neural Networks configuration for all the networks used in the program.
	No longer in use
	
6- Helper.lua
	Active file. Contains helper functions that train, create and test Neural Networks under both Paticipants and Reference User label

7-RandomUploadByParticipant.lua
	Main file that conducts several rounds over all partcipants in which particpants are randomly chosen to interact with the server.
	1- Has multiple rounds
	2- Each round iterates over all the participants and randomly chooses which of the participants will interact with the server. The 		chosen particpants will train, upload and download to and from the server over multiples epoch
	3- At the end of the round, we will print the result of the confusion matrix of the Reference User and how many participants 			interacted with ther Server.
