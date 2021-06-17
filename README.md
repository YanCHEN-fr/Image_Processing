Image Processing

*Author : Yan CHEN*

This is a course (ROB317) project of ENSTA Paris robotics, including 2 parts.

In this project, the paper[1] has been reproduced.

## 1. Detection and Matching of feature point

* ORB

  ![ORB](./README.assets/ORB.png)

  <center>nfeatures = 250, scalarFactor = 1.1, nlevels = 3</center>

* KAZE

  ![KAZE](./README.assets/KAZE.png)

<center>Threshold = 0.001, nOctaves = 4, nOctave- Layers= 4, diffusivity = 2</center>

* FLANN

![FLANN](./README.assets/FLANN.png)

<center>FLANN & KAZE</center>

* RatioTest

![RatioTest](./README.assets/RatioTest.png)

<center>RatioTest & KAZE</center>

* CrossCheck

![CrossCheck](./README.assets/CrossCheck.png)

<center>CrossCheck & KAZE</center>

## 2. Video trimming and indexing

* Colour histogram

* Optical flow and velocity histogram

* Key frame detection
  * Base on optical flow
  
    ![截屏2021-06-16 17.39.27](/Users/apple/Desktop/Image_Processing/README.assets/截屏2021-06-16 17.39.27.png)
  
  <center>First row: the original frames of a consequence. Second row: the 2d histograms corresponding to the joint probability of the components (Vx, Vy) belonging to original frames of a consequence</center>
  
  * Base on frame-by-frame difference
  
  ![fram_by_fram](./README.assets/Frame_nexthsv0065.png)
  
  * Based on the HSV colour model[1]
  
  ![HSV colour model](./README.assets/HSV_colour_model.png)
  
  <center>Key frame detection diagram based on HSV Color Model</center>

## Reference 

[1] Zhi-min XIAO et al. “Shot Segmentation Based on HSV Color Model [J]”. In : Journal of Xiamen University (Natural Science) 5 (2008).