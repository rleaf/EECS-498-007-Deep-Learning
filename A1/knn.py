"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
import statistics


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from knn.py!')


def compute_distances_two_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses a naive set of nested loops over the training and
  test data.

  The input data may have any number of dimensions -- for example this function
  should be able to compute nearest neighbor between vectors, in which case
  the inputs will have shape (num_{train, test}, D); it should alse be able to
  compute nearest neighbors between images, where the inputs will have shape
  (num_{train, test}, C, H, W). More generally, the inputs will have shape
  (num_{train, test}, D1, D2, ..., Dn); you should flatten each element
  of shape (D1, D2, ..., Dn) into a vector of shape (D1 * D2 * ... * Dn) before
  computing distances.

  The input tensors should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point. It should have the same dtype as x_train.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  print('num_train', num_train)
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  print('dists shape', dists.shape)
  ##############################################################################
  # TODO: Implement this function using a pair of nested loops over the        #
  # training data and the test data.                                           #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code

  # Instead of mutating new tensor, reshape old tensor to desired dimensions
  # Flattened tensors
  # train = x_train.view(num_train, x_train[1].view(1, -1).shape[1])
  # test = x_test.view(num_test, x_test[1].view(1, -1).shape[1])
  train = x_train.flatten(1)
  test = x_test.flatten(1)
  print('train shape:', train.shape)
  print('test shape:', test.shape)

  for i in range(num_test):
    for j in range(num_train):
      dists[j, i] = torch.sqrt(torch.sum(torch.square(train[j] - test[i])))

  # print(dists[0:2])
  # print('waffles', torch.sqrt(torch.sum(torch.square(train[1] - test[0]))))
  print('poggers_two', dists[30, 89])
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_one_loop(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses only a single loop over the training data.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.

  Inputs:
  - x_train: Torch tensor of shape (num_train, D1, D2, ...)
  - x_test: Torch tensor of shape (num_test, D1, D2, ...)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using only a single loop over x_train.       #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  # Flatten
  # train = x_train.view(num_train, x_train[1].view(1, -1).shape[1])
  # test = x_test.view(num_test, x_test[1].view(1, -1).shape[1])
  train = x_train.flatten(1)
  test = x_test.flatten(1)
  print('d', train.shape, test.shape)

  # 1 Loop
  for i in range(num_test):
    dists[:,i] = torch.sqrt(torch.sum(torch.square(train-test[i]), 1))
  
  print('hmm', torch.sum(torch.square(train-test[1]), 1).shape)
  # print('hmm2', dists[:,1])

  print('poggers_one', dists[30, 89])
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def compute_distances_no_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation should not use any Python loops. For memory-efficiency,
  it also should not create any large intermediate tensors; in particular you
  should not create any intermediate tensors with O(num_train*num_test)
  elements.

  Similar to compute_distances_two_loops, this should be able to handle inputs
  with any number of dimensions. The inputs should not be modified.

  NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
  `torch.cdist`, or their instance method variants x.norm / x.dist / x.cdist.
  You may not use any functions from torch.nn or torch.nn.functional.
  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function without using any explicit loops and without #
  # creating any intermediate tensors with O(num_train * num_test) elements.   #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  #                                                                            #
  # HINT: Try to formulate the Euclidean distance using two broadcast sums     #
  #       and a matrix multiply.                                               #
  ##############################################################################
  # Replace "pass" statement with your code
  # Find Euclidean distance between points
  # train = x_train.view(num_train, x_train[1].view(1, -1).shape[1])
  # test = x_test.view(num_test, x_test[1].view(1, -1).shape[1])
  train = x_train.flatten(1)
  test = x_test.flatten(1)

  # Euclidean = sqrt(sum(square(a - b)))
  # Can rewrite as sqrt(sum(a^2 + b^2 - 2ab))
  train_sq = torch.square(train)
  test_sq = torch.square(test)

  train_sum_sq = torch.sum(train_sq, 1)
  test_sum_sq = torch.sum(test_sq, 1)

  mul = torch.matmul(train, test.transpose(0, 1))
  print('train', train.shape)
  print('test', test.shape)
  print('corgi', mul.shape)

  print('train', train_sum_sq.reshape(-1, 1).shape)
  print('test', test_sum_sq.reshape(1, -1).shape)

  # Reshape enables proper broadcasting, reshapes train to a [100, 1] column vector and test to a [1, 100],
  # row vector. This forces broadcastig to match dimensions of prior distance measuring (previous functions)
  dists = torch.sqrt(train_sum_sq.reshape(-1, 1) + test_sum_sq.reshape(1, -1) - 2*mul)

  print('dists shape:', dists.shape) 
  # print('poggers_no', dists[30, 89])

  # Populate dists tensor with Euclidean points

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists


def predict_labels(dists, y_train, k=1):
  """
  Given distances between all pairs of training and test samples, predict a
  label for each test sample by taking a **majority vote** among its k nearest
  neighbors in the training set.

  In the event of a tie, this function **should** return the smallest label. For
  example, if k=5 and the 5 nearest neighbors to a test example have labels
  [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes), so
  we should return 1 since it is the smallest label.

  This function should not modify any of its inputs.

  Inputs:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  - y_train: Torch tensor of shape (num_train,) giving labels for all training
    samples. Each label is an integer in the range [0, num_classes - 1]
  - k: The number of nearest neighbors to use for classification.

  Returns:
  - y_pred: A torch int64 tensor of shape (num_test,) giving predicted labels
    for the test data, where y_pred[j] is the predicted label for the jth test
    example. Each label should be an integer in the range [0, num_classes - 1].
  """
  num_train, num_test = dists.shape
  y_pred = torch.zeros(num_test, dtype=torch.int64)
  ##############################################################################
  # TODO: Implement this function. You may use an explicit loop over the test  #
  # samples. Hint: Look up the function torch.topk                             #
  ##############################################################################
  # Replace "pass" statement with your code
  m = []
  for i in range(num_test):
    # Finds location in column[i] of k lowest values
    x = torch.topk(dists[:,i], k, largest=False).indices
    print('x',i,  x)
    # Associates each lowest location with the labels
    m = y_train[x[0:k]]
    print('m',i, m)
    # Populates ith element in y_pred with the most frequent/lowest value in m
    # torch.bincount() is a little tricky. Returns the frequency of occurences of ith element and
    # and when called, returns the value it is corresponding too.
    # torch.argmax will ironically return the lowest most frequent value (if there are numerous)
    # because it returns the *first* max value, which torch.bincount sorts numerically (so it will always be the lowest)
    y_pred[i] = torch.argmax(torch.bincount(m))
  print('y_pred', i, y_pred)
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return y_pred


class KnnClassifier:
  def __init__(self, x_train, y_train):
    """
    Create a new K-Nearest Neighbor classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Inputs:
    - x_train: Torch tensor of shape (num_train, C, H, W) giving training data
    - y_train: int64 torch tensor of shape (num_train,) giving training labels
    """
    ###########################################################################
    # TODO: Implement the initializer for this class. It should perform no    #
    # computation and simply memorize the training data.                      #
    ###########################################################################
    # Replace "pass" statement with your code
    self.x_train = x_train
    self.y_train = y_train
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

  def predict(self, x_test, k=1):
    """
    Make predictions using the classifier.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - k: The number of neighbors to use for predictions

    Returns:
    - y_test_pred: Torch tensor of shape (num_test,) giving predicted labels
      for the test samples.
    """
    y_test_pred = None
    ###########################################################################
    # TODO: Implement this method. You should use the functions you wrote     #
    # above for computing distances (use the no-loop variant) and to predict  #
    # output labels.
    ###########################################################################
    # Replace "pass" statement with your code
    y_test_pred = torch.zeros(x_test.shape[0], dtype=torch.int64)

    # Flatten input tensors
    num_train = self.x_train.shape[0]
    num_test = x_test.shape[0]
    x_train_flat = self.x_train.view(num_train, self.x_train[1].view(1, -1).shape[1])
    x_test_flat = x_test.view(num_test, x_test[1].view(1, -1).shape[1])
    # print(x_train_flat.shape)
    # print(x_test_flat.shape)

    # Find Euclidean Distance --> sqrt(sum(square(a-b))) or sqrt(sum(a^2+b^2-2ab))
    train_sq = torch.square(x_train_flat)
    test_sq = torch.square(x_test_flat)

    # print('train dim:', x_train_flat.shape)
    # print('test dim:', x_test_flat.shape)

    # Sums values of row elements so that train & test become 1d tensors
    train_sum_sq = torch.sum(train_sq, 1)
    test_sum_sq = torch.sum(test_sq, 1)

    # print('penguins', x_test_flat.shape, x_train_flat.transpose(0, 1).shape)
    # mul = x_test_flat.matmul(x_train_flat.transpose(0, 1))
    mul = torch.matmul(x_train_flat, x_test_flat.transpose(0, 1))

    # Creating empty tensor to populate with Euclidean distances
    pancakes = torch.zeros(x_train_flat.shape[0], x_test_flat.shape[0]) # [50000, 10000]

    # Populating pancakes with Euclidean distances. No loops required because simply pushing a
    # n x m matrix into another n x m matrix.
    pancakes = torch.sqrt(train_sum_sq.reshape(-1, 1) + test_sum_sq.reshape(1, -1) - 2*mul)
    # print('pancakes shape:', pancakes.shape)

    # Begin labeling
    # print('labels:', self.y_train)
    # print('toad', pancakes.shape)
    # print('toad', pancakes.shape[0])
    for i in range(pancakes.shape[1]):
      y = []
      # Index over each column to find k lowest values
      x = torch.topk(pancakes[:,i], k, largest=False).indices
      y = self.y_train[x[0:k]]

      # If k=1, having right hand side of equality isn't necessary; but on later iterations
      # k=3, and k=5.
      y_test_pred[i] = torch.argmax(torch.bincount(y))
    
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_test_pred

  def check_accuracy(self, x_test, y_test, k=1, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - k: The number of neighbors to use for prediction
    - quiet: If True, don't print a message.

    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test, k=k)
    num_samples = x_test.shape[0]
    num_correct = (y_test == y_test_pred).sum().item()
    accuracy = 100.0 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
           f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy


def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
  """
  Perform cross-validation for KnnClassifier.

  Inputs:
  - x_train: Tensor of shape (num_train, C, H, W) giving all training data
  - y_train: int64 tensor of shape (num_train,) giving labels for training data
  - num_folds: Integer giving the number of folds to use
  - k_choices: List of integers giving the values of k to try

  Returns:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.
  """
  if k_choices is None:
    # Use default values
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    # k_choices = [1, 3, 5]

  # First we divide the training data into num_folds equally-sized folds.
  x_train_folds = []
  y_train_folds = []
  ##############################################################################
  # TODO: Split the training data and images into folds. After splitting,      #
  # x_train_folds and y_train_folds should be lists of length num_folds, where #
  # y_train_folds[i] is the label vector for images in x_train_folds[i].       #
  # Hint: torch.chunk                                                          #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_flat = x_train.view(x_train.shape[0], x_train[1].view(1, -1).shape[1])
  # print('waffles', x_train_flat.shape)

  x_train_folds = torch.chunk(x_train_flat, num_folds, dim=0)
  y_train_folds = torch.chunk(y_train, num_folds, dim=0)
  print('ad', y_train_folds[0].shape)
  # x = torch.chunk(x_train_flat, num_folds, dim=0)
  # y = torch.chunk(y_train, num_folds, dim=0)

  # for i in range(num_folds):
  #   x_train_folds.append(x[i])
  #   y_train_folds.append(y[i])
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracies we found when trying KnnClassifiers that use k neighbors.
  k_to_accuracies = {}

  ##############################################################################
  # TODO: Perform cross-validation to find the best value of k. For each value #
  # of k in k_choices, run the k-nearest-neighbor algorithm num_folds times;   #
  # in each case you'll use all but one fold as training data, and use the     #
  # last fold as a validation set. Store the accuracies for all folds and all  #
  # values in k in k_to_accuracies.   HINT: torch.cat                          #
  ##############################################################################
  # Replace "pass" statement with your code

  for k in k_choices:
    for folds in range(num_folds):
      x_valid = x_train_folds[folds]
      y_valid = y_train_folds[folds]

      x_traink = torch.cat(x_train_folds[:folds] + x_train_folds[folds + 1:])
      y_traink = torch.cat(y_train_folds[:folds] + y_train_folds[folds + 1:])

      knn = KnnClassifier(x_traink, y_traink)

      accuracy = knn.check_accuracy(x_valid, y_valid, k=k)
      k_to_accuracies.setdefault(k, []).append(accuracy)
      

  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################
  print('waffles', k_to_accuracies)
  return k_to_accuracies


def knn_get_best_k(k_to_accuracies):
  """
  Select the best value for k, from the cross-validation result from
  knn_cross_validate. If there are multiple k's available, then you SHOULD
  choose the smallest k among all possible answer.

  Inputs:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.

  Returns:
  - best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info
  """
  best_k = 0
  ##############################################################################
  # TODO: Use the results of cross-validation stored in k_to_accuracies to     #
  # choose the value of k, and store the result in best_k. You should choose   #
  # the value of k that has the highest mean accuracy accross all folds.       #
  ##############################################################################
  # Replace "pass" statement with your code
  keys = [k for k in k_to_accuracies.keys()]
  values = [v for v in k_to_accuracies.values()]
  # print('keys', keys)
  # print('values', values)

  max_avg = torch.argmax(torch.mean(torch.tensor(values), dim=1))
  best_k = keys[max_avg]
  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################
  return best_k
