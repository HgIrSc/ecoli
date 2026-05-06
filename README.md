# 形态学方法提取大肠杆菌二维轨迹

使用形态学方法从相衬显微影像中提取在焦平面附近大肠杆菌的二维轨迹

首先对宽度 200 为帧的窗口内的图像计算平均，以此为背景从每帧中扣除，从
而消除不均匀光照与静止背景。截取图片中亮度值在平均值以下的部分。随后使
用最大熵方法[1]对每帧图像进行二值化。对二值化图像进行连通区域分析，提取
出每个连通区域的面积、坐标。随后对连通区域进行过滤，取面积在 10–50 pixel2
内的连通区域，认为这些连通区域代表大肠杆菌。如此可以消除在焦平面之外细
菌的影响。最后使用 Crocker 等人的方法[2] 将离散的坐标点连接，得到大肠杆
菌的二维运动轨迹。

# Extract Trajectories of _E.coli_ Using Morphological Methods

Extract two-dimensional trajectories of _E. coli_ near focal plane using morphological methods.

First, the average of the images within a window with a width of 200 frames
is calculated, and this is used as the background to be subtracted from each frame,
thereby eliminating uneven lighting and static background.
The portion of the image with a brightness value below the average value is cropped.
Then, the maximum entropy method [1] is used to binarize each frame image.
Connectivity analysis is performed on the binarized image to extract the area
and coordinates of each connected region.
Then, the connected regions are filtered, and connected regions with an area of
10–50 pixel x pixel are taken as representing E. coli. This can eliminate the
influence of bacteria outside the focal plane. Finally, the discrete coordinate
points are connected using the method of Crocker et al. [2] to obtain the
two-dimensional motion trajectory of E. coli.

[1] Kapur J, Sahoo P, Wong A. A new method for gray-level picture thresholding
using the entropy of the histogram[J/OL]. Computer Vision, Graphics, and Image
Processing, 1985, 29(3): 273-285 [2026-04-09]. https://linkinghub.elsevier.com
/retrieve/pii/0734189X85901252. DOI: 10.1016/0734-189X(85)90125-2.

[2] Crocker J C, Grier D G. Methods of Digital Video Microscopy for Colloidal
Studies[J/OL]. Journal of Colloid and Interface Science, 1996, 179(1): 298-310
[2025-05-16]. https://linkinghub.elsevier.com/retrieve/pii/S0021979796902179.
DOI: 10.1006/jcis.1996.0217
