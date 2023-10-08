# Improving Transformer-based Image Matching by Cascaded Capturing Spatially Informative Keypoints (ICCV2023)

[Chenjie Cao](https://ewrfcas.github.io/),
[Yanwei Fu](http://yanweifu.github.io/)


[Project Page](https://ewrfcas.github.io/CasMTR/)

## Abstract

Learning robust local image feature matching is a fundamental
low-level vision task, which has been widely explored
in the past few years. Recently, detector-free local feature
matchers based on transformers have shown promising results,
which largely outperform pure Convolutional Neural
Network (CNN) based ones. But correlations produced by
transformer-based methods are spatially limited to the center
of source views’ coarse patches, because of the costly
attention learning. In this work, we rethink this issue and
find that such matching formulation degrades pose estimation,
especially for low-resolution images. So we propose a
transformer-based cascade matching model – Cascade feature
Matching TRansformer (CasMTR), to efficiently learn
dense feature correlations, which allows us to choose more
reliable matching pairs for the relative pose estimation. Instead
of re-training a new detector, we use a simple yet effective
Non-Maximum Suppression (NMS) post-process to filter
keypoints through the confidence map, and largely improve
the matching precision. CasMTR achieves state-of-the-art
performance in indoor and outdoor pose estimation as well
as visual localization. Moreover, thorough ablations show
the efficacy of the proposed components and techniques.

