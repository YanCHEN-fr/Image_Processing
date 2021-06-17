Image Processing

*Author : Yan CHEN*

This is a course (ROB317) project of ENSTA Paris robotics, including 2 parts.

In this project, the paper[1] has been reproduced.

## 1. Detection and Matching of feature point

* ORB

  <p align="center"><img width="60%" src="./README.assets/ORB.png"></p>

  <center>nfeatures = 250, scalarFactor = 1.1, nlevels = 3</center>

* KAZE

  <p align="center"><img width="60%" src="./README.assets/KAZE.png"></p>

<center>Threshold = 0.001, nOctaves = 4, nOctave- Layers= 4, diffusivity = 2</center>

* FLANN

<p align="center"><img width="60%" src="./README.assets/FLANN.png"></p>

<center>FLANN & KAZE</center>

* RatioTest

  <p align="center"><img width="60%" src="./README.assets/RatioTest.png"></p>

<center>RatioTest & KAZE</center>

* CrossCheck

  <p align="center"><img width="60%" src="./README.assets/CrossCheck.png"></p>

<center>CrossCheck & KAZE</center>

## 2. Video trimming and indexing

* Colour histogram

* Optical flow and velocity histogram

* Key frame detection
  * Base on optical flow
  
    <p align="center"><img width="80%" src="./README.assets/optical_flow.png"></p>
  
  <center>First row: the original frames of a consequence. Second row: the 2d histograms corresponding to the joint probability of the components (Vx, Vy) belonging to original frames of a consequence</center>
  
  * Base on frame-by-frame difference
  
    <p align="center"><img width="60%" src="./README.assets/Frame_nexthsv0065.png"></p>
  
  * Based on the HSV colour model[1]
  
    <p align="center"><img width="60%" src="./README.assets/HSV_colour_model.png"></p>
  
  <p align="center">Key frame detection diagram based on HSV Color Model</p>

## Reference 

[1] Zhi-min XIAO et al. “Shot Segmentation Based on HSV Color Model [J]”. In : Journal of Xiamen University (Natural Science) 5 (2008).

