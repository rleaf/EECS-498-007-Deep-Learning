import time
import math
from numpy import dtype
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  A, wh = anc.shape
  B, H, W, xy = grid.shape

  anchors = torch.zeros((B, A, H, W, 4), dtype=anc.dtype, device=anc.device)
  anc = anc/2

  # Able to produce anchors by broadcasting grid with anc.
  # To broadcast, have to make dimensions communicable. Ex below.
  # https://pytorch.org/docs/stable/notes/broadcasting.html
  # (3, 1, 7, 7, 2)
  # (1, 9, 1, 1, 2)

  # print('pre', grid.shape)
  # print('pre', anc.shape)
  grid = grid.unsqueeze(1)
  anc = anc.unsqueeze(1).unsqueeze(1).unsqueeze(0)
  # Another way of doing above
  # anc = anc.reshape(1, A, 1, 1, wh)

  # First two positions (x_tl, y_tl)
  anchors[:,:,:,:,:2] = grid - anc

  # Last two positions (x_br, y_br)
  anchors[:,:,:,:,2:] = grid + anc
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code

  proposals = torch.zeros_like(anchors)
  _proposals = torch.zeros_like(anchors)
  centers = torch.zeros_like(anchors)

  # Paramaterize to centers
  # 
  # anchors = (xtl, ytl, xbr, ybr)
  # centers = (xc, yc, w, h)

  centers[:, :, :, :, 0] = (anchors[:, :, :, :, 2] + anchors[:, :, :, :, 0]) / 2
  centers[:, :, :, :, 1] = (anchors[:, :, :, :, 1] + anchors[:, :, :, :, 3]) / 2
  centers[:, :, :, :, 2] = anchors[:, :, :, :, 2] - anchors[:, :, :, :, 0]
  centers[:, :, :, :, 3] = anchors[:, :, :, :, 3] - anchors[:, :, :, :, 1]

  if (method == 'YOLO'):
    _proposals[:, :, :, :, 0] = centers[:, :, :, :, 0] + offsets[:, :, :, :, 0]
    _proposals[:, :, :, :, 1] = centers[:, :, :, :, 1] + offsets[:, :, :, :, 1]
    _proposals[:, :, :, :, 2] = centers[:, :, :, :, 2] * torch.exp(offsets[:, :, :, :, 2])
    _proposals[:, :, :, :, 3] = centers[:, :, :, :, 3] * torch.exp(offsets[:, :, :, :, 3])
  
  if (method == 'FasterRCNN'):
    _proposals[:, :, :, :, 0] = centers[:, :, :, :, 0] + (centers[:, :, :, :, 2] * offsets[:, :, :, :, 0])
    _proposals[:, :, :, :, 1] = centers[:, :, :, :, 1] + (centers[:, :, :, :, 3] * offsets[:, :, :, :, 1])
    _proposals[:, :, :, :, 2] = centers[:, :, :, :, 2] * torch.exp(offsets[:, :, :, :, 2])
    _proposals[:, :, :, :, 3] = centers[:, :, :, :, 3] * torch.exp(offsets[:, :, :, :, 3])
  
  # Go back to original anchor paramaterization
  # 
  proposals[:, :, :, :, 0] = _proposals[:, :, :, :, 0] - _proposals[:, :, :, :, 2] / 2
  proposals[:, :, :, :, 2] = _proposals[:, :, :, :, 0] + _proposals[:, :, :, :, 2] / 2
  proposals[:, :, :, :, 1] = _proposals[:, :, :, :, 1] - _proposals[:, :, :, :, 3] / 2
  proposals[:, :, :, :, 3] = _proposals[:, :, :, :, 1] + _proposals[:, :, :, :, 3] / 2

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  B, A, H, W, _ = proposals.shape
  _, N, _ = bboxes.shape

  iou_mat = torch.zeros((B, A*H*W, N), dtype=proposals.dtype, device=proposals.device)
  proposals = proposals.reshape(B,A*H*W,4).unsqueeze(2).repeat(1, 1, N, 1)
  bboxes = bboxes.unsqueeze(1)

  # (xtl, ytl, xbr, ybr)
  # Broadcast every region proposals against the ground truths to output a (B, 441, 40) tensor. 
  # Due to reshaping of both prop & bbox tensor, we are indexing for the max value of each proposal
  # to every bounding box, then move onto the next propsal and do the same until the end.
  x_tl = torch.max(proposals[:, :, :, 0], bboxes[:, :, :, 0])
  y_tl = torch.max(proposals[:, :, :, 1], bboxes[:, :, :, 1])
  x_br = torch.min(proposals[:, :, :, 2], bboxes[:, :, :, 2])
  y_br = torch.min(proposals[:, :, :, 3], bboxes[:, :, :, 3])

  aoi = torch.clamp(x_br - x_tl, min=0) * torch.clamp(y_br - y_tl, min=0)
  aop = (proposals[:, :, :, 2] - proposals[:, :, :, 0]) * (proposals[:, :, :, 3] - proposals[:, :, :, 1])
  aob = (bboxes[:, :, :, 2] - bboxes[:, :, :, 0]) * (bboxes[:, :, :, 3] - bboxes[:, :, :, 1])

  aou = aop + aob - aoi
  iou_mat = aoi / aou
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_dim, hidden_dim, 1),
      nn.Dropout(drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(hidden_dim, 5 * self.num_anchors + self.num_classes, 1)
    )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          # 
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # torch.Size([3, 65, 7, 7])
    anchor_features=self.pred_layer(features)
    # split features into conf_offsets_package and cls_scores_package
    # torch.Size([3, 45, 7, 7])
    conf_offsets_package = anchor_features[:,:5*self.num_anchors,:,:]
    B,_,H,W = conf_offsets_package.shape 
    # Segregate anchors & (transformations, objectness score)
    conf_offsets_package = conf_offsets_package.reshape(B,self.num_anchors,5,H,W)
    # torch.Size([3, 20, 7, 7])
    cls_scores_package = anchor_features[:,5*self.num_anchors:,:,:]

    # conf_offsets_package torch.Size([Obj score, t^x, t^y, w, h])
    # apply sigmoid fucntion to conf[0], t^x[1], t^y[2]
    conf_offsets_package[:,:,0,:,:] = torch.sigmoid(conf_offsets_package[:,:,0,:,:].clone())
    conf_offsets_package[:,:,1:3,:,:] = torch.sigmoid(conf_offsets_package[:,:,1:3,:,:].clone())-0.5

    # testing stage
    if pos_anchor_idx is None and neg_anchor_idx is None:
      conf_scores, offsets, class_scores = conf_offsets_package[:,:,0,:,:],conf_offsets_package[:,:,1:,:,:],cls_scores_package
    else:
      # training stage
      extracted_conf_offsets_package_pos = self._extract_anchor_data(conf_offsets_package,pos_anchor_idx)
      extracted_conf_offsets_package_neg = self._extract_anchor_data(conf_offsets_package,neg_anchor_idx)
      conf_scores = torch.cat((extracted_conf_offsets_package_pos,extracted_conf_offsets_package_neg), dim=0)[:,0:1]
      offsets = extracted_conf_offsets_package_pos[:,1:]
      
      class_scores = self._extract_class_scores(cls_scores_package,pos_anchor_idx)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    feat = self.feat_extractor(images)
    grid_list = GenerateGrid(images.shape[0])
    anc_list = GenerateAnchor(self.anchor_list.to(images.device), grid_list.to(images.device))
    iou_mat = IoU(anc_list, bboxes)
    # total anchors in an image (9 * 7 * 7)
    anc_per_img = torch.prod(torch.tensor(anc_list.shape[1:-1]))

    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
      _, _ = ReferenceOnActivatedAnchors(anc_list, bboxes, grid_list, iou_mat, neg_thresh=0.2)

    conf_scores, offsets, class_scores = self.pred_network(feat, activated_anc_ind, negative_anc_ind)

    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    cls_loss = ObjectClassification(class_scores, GT_class, images.shape[0], anc_per_img, activated_anc_ind)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    feat = self.feat_extractor(images)

    # Generate anchor positions corresponding to input
    grid_list = GenerateGrid(images.shape[0])
    anc_list = GenerateAnchor(self.anchor_list.to(images.device), grid_list.to(images.device)) # (B, A, H', W', 4)

    # Forward pass through prediction network
    # conf_scores torch.Size([B, A, H', W'])
    # offsets torch.Size([B, A, 4, H', W'])
    # class_scores torch.Size([B, C, H', W'])
    conf_scores, offsets, class_scores = self.pred_network(feat)

    batch_size = images.shape[0]
    offsets = offsets.permute(0, 1, 3, 4, 2)
    proposals = GenerateProposal(anc_list, offsets) # (B, A, H', W', 4)

    # flat_proposals = proposals.reshape(-1, 4) # (B*A*H'*W', 4)
    flat_proposals = proposals.reshape(batch_size, -1, 4) # (B, A*H'*W', 4)
    flat_conf_scores = conf_scores.reshape(batch_size, -1) # (B, A*H'*W')

    # Conf score thresholding
    mask = flat_conf_scores > thresh # (B, A*H'*W')

    _, max_cls = class_scores.max(1) # (B, H', W')
    max_cls = max_cls.reshape(batch_size, -1) # (B, H'*W')
    max_cls = max_cls.unsqueeze(1).repeat(1, proposals.shape[1], 1).reshape(batch_size, -1) # (B, A*H'*W')

    for i in range(batch_size):
      thresh_proposals = flat_proposals[i][mask[i]] # ith image (*, 4)
      thresh_conf_scores = flat_conf_scores[i][mask[i]] # ith image (*, 1)
      i_class_scores = max_cls[i][mask[i]] # ith image (*, 1)
      # Get nms indices for ith image
      nms_keep = nms(thresh_proposals, thresh_conf_scores, nms_thresh)

      idx = nms_keep.long()
      final_proposals.append(thresh_proposals[idx])
      final_conf_scores.append(thresh_conf_scores[idx].unsqueeze(1))
      final_class.append(i_class_scores[idx].unsqueeze(1))
      
    # print(final_proposals[1].shape)
    # print(final_conf_scores[1].shape)
    # print(final_class[1].shape)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  N, _ = boxes.shape

  keep = []
  sorted, indices = torch.sort(scores, descending=True)
  
  while(len(indices) > 0):
    # Appending values to cleaned tensor
    max_idx = indices[0]
    keep.append(max_idx)
    if(topk and len(keep) == topk):
      return torch.tensor(keep, dtype=boxes.dtype, device=boxes.device)
    
    # Intersection / Union
    # Batch calculate between [max_idx] and all other boxes
    x1 = torch.max(boxes[max_idx][0], boxes[indices][:, 0])
    y1 = torch.max(boxes[max_idx][1], boxes[indices][:, 1])
    x2 = torch.min(boxes[max_idx][2], boxes[indices][:, 2])
    y2 = torch.min(boxes[max_idx][3], boxes[indices][:, 3])

    aoi = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    ao1 = (boxes[max_idx][2] - boxes[max_idx][0]) * (boxes[max_idx][3] - boxes[max_idx][1])
    ao2 = (boxes[indices][:, 2] - boxes[indices][:, 0]) * (boxes[indices][:, 3] - boxes[indices][:, 1])

    iou = aoi / (ao1 + ao2 - aoi)
    # decriment
    wtf = iou <= iou_threshold
    indices = indices[wtf]
  keep = torch.tensor(keep, dtype=boxes.dtype, device=boxes.device)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

