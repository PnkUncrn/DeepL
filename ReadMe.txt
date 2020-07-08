Summary

existing distributed learning approaches are vulnerable to attacks where a malicious user can use the the shared neural
network parameters to recreate the private data from other users.
We propose a distributed deep learning algorithm that allows a user to improve its
deep-learning model while preserving its privacy from such attacks. Specifically, our
approach focuses on protecting the privacy of a single user by limiting the number of times
other users can download and upload parameters from the main deep neural network. By
doing so, our approach limits ability of the attackers to recreate private data samples from
the reference user while maintaining a highly accurate deep neural network.
Our approach is flexible and can be adapted to work with any deep neural network
architectures. We conduct extensive experiments to verify the proposed approach. We
observe that the trained neural network can achieve an accuracy of 95.18%, while
protecting the privacy of the reference user by preventing it from sharing both its private
data and deep neural network parameters with the server or other users.




The following is a short description of all the files in the DeepL directory.
This is a working copy

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
8- Deep Learning Thesis Draft.tex
	1-Running draft of the final report
	
