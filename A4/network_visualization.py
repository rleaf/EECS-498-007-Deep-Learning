"""
Implements a network visualization in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

# import os
import torch
# import torchvision
# import torchvision.transforms as T
# import random
# import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch._C import device
from a4_helper import *


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from network_visualization.py!')

def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.

  Input:
  - X: Input images; Tensor of shape (N, 3, H, W)
  - y: Labels for X; LongTensor of shape (N,)
  - model: A pretrained CNN that will be used to compute the saliency map.

  Returns:
  - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
  images.
  """
  # Make input tensor require gradient
  X.requires_grad_()
  
  saliency = None
  ##############################################################################
  # TODO: Implement this function. Perform a forward and backward pass through #
  # the model to compute the gradient of the correct class score with respect  #
  # to each input image. You first want to compute the loss over the correct   #
  # scores (we'll combine losses across a batch by summing), and then compute  #
  # the gradients with a backward pass.                                        #
  # Hint: X.grad.data stores the gradients                                     #
  ##############################################################################
  # Replace "pass" statement with your code

  scores = model(X)

  # print('X', X.shape)
  # print('y', y.shape)
  # print('scores', scores.shape)
  correctScore = torch.zeros_like(y, dtype=torch.float32)

  for i in range(y.shape[0]):
    correctScore[i] = scores[i, y[i]]
  # print('y', y)
  # print('y2', correctScore)
  # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
  # scores = (scores.gather(1, y.view(-1, 1)).squeeze())
  #backward pass
  correctScore.backward(torch.ones(X.shape[0], device=X.device))
  # print('toaders', X.grad.data.shape)
  # scores.backward(torch.ones(X.shape[0], device=X.device))
  #saliency
  print('aed', X.grad.data.shape)
  # Says to take the max value of the 3 color channels per pixel, but could summing work too?
  # saliency = torch.sum(X.grad.data.abs(), dim=1)
  saliency, _ = torch.max(X.grad.data.abs(), dim=1)
  ##############################################################################
  #               END OF YOUR CODE                                             #
  ##############################################################################
  return saliency

def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
  """
  Generate an adversarial attack that is close to X, but that the model classifies
  as target_y.

  Inputs:
  - X: Input image; Tensor of shape (1, 3, 224, 224)
  - target_y: An integer in the range [0, 1000)
  - model: A pretrained CNN
  - max_iter: Upper bound on number of iteration to perform
  - verbose: If True, it prints the pogress (you can use this flag for debugging)

  Returns:
  - X_adv: An image that is close to X, but that is classifed as target_y
  by the model.
  """
  # Initialize our adversarial attack to the input image, and make it require gradient
  X_adv = X.clone()
  X_adv = X_adv.requires_grad_()
  
  learning_rate = 1
  ##############################################################################
  # TODO: Generate an adversarial attack X_adv that the model will classify    #
  # as the class target_y. You should perform gradient ascent on the score     #
  # of the target class, stopping when the model is fooled.                    #
  # When computing an update step, first normalize the gradient:               #
  #   dX = learning_rate * g / ||g||_2                                         #
  #                                                                            #
  # You should write a training loop.                                          #
  #                                                                            #
  # HINT: For most examples, you should be able to generate an adversarial     #
  # attack in fewer than 100 iterations of gradient ascent.                    #
  # You can print your progress over iterations to check your algorithm.       #
  ##############################################################################
  # Replace "pass" statement with your code
  hmm = 0
  for i in range(max_iter):
    scores = model(X_adv)
    _, ypred = torch.max(scores, dim=1)
    if(ypred == target_y):
      break
    
    scores[:, target_y].backward()

    with torch.no_grad():
      # don't need X_adv.grad.data ?
      X_adv += learning_rate * (X_adv.grad / torch.norm(X_adv.grad))
      # X_adv.grad.data.zero_()
    # print('dta', X_adv.shape)
    # print(X_adv.grad.data)  
    hmm += 1
  print(f'{hmm} iterations')
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the 
    score of target_y under a pretrained model.
  
    Inputs:
    - img: random image with jittering as a PyTorch tensor  
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    scores = model(img)

    I = scores[:, target_y] - (l2_reg * (torch.norm(img)**2))
    I.backward()
    
    with torch.no_grad():
      img += learning_rate * (img.grad / torch.norm(img.grad))
      img.grad.zero_()
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
