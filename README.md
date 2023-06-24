# AI Image Inpainting & Modification Detection
This repository contains a comprehensive project on automatic image inpainting and detection of AI-based image modifications undertaken as the final project for the Summer term 2023 course [High-Level Computer Vision](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/teaching/courses-1/ss-2023-high-level-computer-vision)  at Saarland University.

## Project Description
We aim to tackle the growing issue of misinformation spread through AI-modified images. This project comprises two key components: an automatic image inpainting model and a detector for AI-based image alterations.

*Automatic Inpainting*: We propose an automatic image inpainting system by utilizing deep learning techniques and leveraging state-of-the-art models like the Inpaint Anything model by Yu et al. Our model segments images selects specific objects, and then modifies these objects using inpainting techniques. This process demonstrates how real photos can be maliciously altered without human supervision, highlighting the potential dangers of image manipulation.

*Modification Detection*: To counterbalance the threat posed by our inpainting model, we are also developing a model capable of detecting and locating these modifications. We train this model using a dataset constructed by our automatic inpainting pipeline, demonstrating its defensive capabilities against AI-manipulated images.

## Objectives
Our primary goals for this project are:

To create an automatic inpainting pipeline that modifies images in subtle and misleading ways.
To build a robust model that can detect and locate these modifications based on a dataset that we construct.

## Methodology
Our inpainting pipeline follows a three-step process:

*Segmentation*: We use the Segment Anything Model (SAM) to compute segmentation masks for each object in an image.
Prompt Generation: Image captioning models generate captions for the input image, which are then transformed into malicious prompts using a text-to-text model.

*Inpainting Application*: The original image, one of the segmentation masks, and the malicious prompt are passed to an inpainting model based on stable diffusion, which paints a new object within the bounds of the segmentation mask.
For our detector, we apply the inpainting pipeline to an image dataset to create a dataset of modified images and corresponding segmentation masks. The detector is then trained on this dataset to detect inpainting-based image modifications.
