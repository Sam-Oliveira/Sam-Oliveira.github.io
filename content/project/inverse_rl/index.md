---
title: Inverse Reinforcement Learning using Generative Planning Models in Trajectory Space
summary: MSc Thesis in Ilija Bogunovic's Group, as part of the MSc in Machine Learning at UCL.
tags:
  - RL
date: '2024-10-01'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: Guided trajectory-level diffusion schematic.
  focal_point: Smart

links:
url_code: 
url_pdf: uploads/adl.pdf
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---

Self-supervised learning approaches have gained significant popularity in recent
years, specifically when used for the pre-training of very large foundation models,
both in NLP and Computer Vision. These methods allow for pretraining on very large amounts of data (for which collection of labels would be
impractical), resulting in models which are capable of learning richer embedded
representations. These models can then be fine-tuned to specific tasks using
significantly less data than if trained from scratch.

In computer vision, these methods are mostly divided in invariance-based
methods and generative methods. The former are based on training an encoder-like network to produce similar embeddings for images of the
same scene, but with different views. This is the idea behind contrastive learning. The embeddings will thus provide a representation with high semantic
value. Generative methods (in which masked autoencoders (MAES) are
included) are based on corrupting portions of the input images, and learning
to predict these corrupted portions. In doing so, the model learns meaningful
representations, albeit of a lower level than contrastive methods. Recent work
has proposed a third approach, named IJEPA, which also does not rely on
data augmentation, but differs from MAEs by computing the loss in embedding
space, thus promoting meaningful embedded representations.

A natural question that arises from the pre-training of large models is how
the distribution of the pre-training data affects the model’s performance on
downstream tasks. In this paper, we hypothesise that pre-training on data
from a similar distribution to the fine-tuning data should result in better model
performance. We implement a Masked Autoencoder, and pre-train it on the
MS COCO dataset. This pre-trained model is then fine-tuned to perform
image segmentation on the Oxford Pet dataset. Different splits of the MS
COCO pre-training dataset are attempted so as to study the impact of the pretraining data distribution on the down-stream segmentation results.