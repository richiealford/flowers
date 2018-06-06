import torch 
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models

def makeModel():

	model = models.densenet121(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False

	classifier = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(1024, 500)),
                                          ('relu', nn.ReLU()),
                                          ('fc2', nn.Linear(500, 102)),
                                          ('output', nn.LogSoftmax(dim=1))
                                          ]))
	model.classifier = classifier
	return model
	

def train(model, trainloader, testloader, criterion, optimizer, epochs, logging=True):
	
	iteration = 0
	for epoch in range(epochs):
		for images, labels in trainloader:
			iteration += 1
			model.train()
			optimizer.zero_grad()
			outputs = model.forward(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
			if iteration % 5 == 0:
				model.eval()
				with torch.no_grad():
					test_loss, accuracy = validation(model, testloader, criterion) 
				
				trainlog(epochs, epoch, iteration, loss, test_loss, accuracy)
	
	return "done" 


def validation(model, testloader, criterion):
	
	accuracy = 0
	test_loss = 0
	
	for images, labels in testloader:
		output = model.forward(images)
		test_loss += criterion(output, labels).item()
		ps = torch.exp(output)
		equality = (labels.data == ps.max(1)[1])
		accuracy += equality.type_as(torch.FloatTensor()).mean()
	
	return test_loss/len(testloader), accuracy/len(testloader) 


def trainlog(epochs, epoch, iteration, loss, test_loss, accuracy):

	print("Epoch: {}/{}.. ".format(epoch, epochs),
		"Iteration: {}..".format(iteration),
		"Training Loss: {}..".format(loss),
		"Testing Loss: {}..".format(test_loss),
		"Accuracy: {}..".format(accuracy))

	return "done"





