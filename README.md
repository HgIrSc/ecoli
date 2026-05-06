# Codes To Extract And Analyze Trajectories of _E.coli_

The extraction of trajectories have six steps:

1. For each image sequence, a background image was calculated by pixel-wise time
  average projection. It was subtracted from each frame to eliminate non-motile
  objects and shading effects. Be implemented in file `subback.py`.
2. Do morphological erosion to reduce the background noise. Be implemented in
  file `denoise.py`.
3. The putative bacterial cells are distinguished from background using the
  maximum entropy thresholding algorithm by Kapur _et al_.
4. The binary images were further processed with the morphological operations,
  imopen and imclose to eliminate any noise caused by segmentation.
5. Find all connected objects in the binary images, and calculate size and
  centroid position of each object.
6. Finally, filter particles using its area.

Global options and parameters are storaged in the json file `config.json`, and
can be loaded to a dictionary by calling function `load_config` from file
`config.py`.
