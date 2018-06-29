# Object-Detection-Models-keras-tesnorflow-
This repository contains the most commonly used object detection models.

# There are many classic object detection models and most state-of-the-art models are optimizations of them.

Model List:
(1) RCNN
(2) Fast RCNN
(3) Faster RCNN
(4) RFCN
(5) Retina Net
(6) SSD
(7) YOLOv3

Description
(1) RCNN ----- Selective Search
    Selective_Search + Crop/Resize + Convolution + FullyConnectedLayer
    
    # Pseudocode
    ROIs = region_proposal(image)
    for ROI in ROIs:
      patch = get_patch(image, ROI) 
      results = detector(pach)

(2) Fast RCNN ----- ROI Pooling
    Convolution + (Selective_Search + ROI Pooling) + Fully_Connected_Layer
    
    # Pseudocode
    feature_maps = process(image)
    ROIs = region_proposal(feature_maps)
    for ROI in ROIs:
        patch = roi_pooling(feature_maps, ROI) 
        results = detector2(patch)

(3) Faster RCNN ----- Anchor + Region Proposal Network
    Convolution + (Anchor + Region Proposal Network) + ROI_Pooling + Fully_Connected_Layer
    
    # Pseudocode
    feature_maps = process(image)
    ROIs = region_proposal(feature_maps)
    for ROI in ROIs:
        patch = roi_pooling(feature_maps, ROI) 
        results = detector2(patch)

(4) RFCN(Region-based Fully Convolutional Network) ----- position-sensitive score map + position-sensitive ROI-pool
    Convolution + (Anchor + Region Proposal Network) + Convolution + Position_Sensitive_ROI_Pool
    
    # Pseudocode
    feature_maps = process(image)
    ROIs = region_proposal(feature_maps)
    score_maps = convolution(feature_maps)
    for ROI in ROIs:
        patch = Position_Sensitive_ROI_Pool(score_maps, ROI) 
        results = VotePooling(patch)

(5) Retina Net ----- Focal Loss
    Convolution + Regional Preposal Network
    feature_maps = process(image)
    results = region_proposal(feature_maps)

(6) SSD
(7) YOLOv3
