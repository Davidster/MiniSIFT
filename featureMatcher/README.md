## Dependencies

  - Pillow
  - opencv

## How to run

- `pip3 install opencv-python Pillow`
- Adjust value of imageTree variable in the main function (near the bottom of the file)
- Run script: `python3 featureMatcher.py`
- View results in out.jpg

Stiching rainierImageTree3 (AllStitched.png) takes about 2 minutes to run on my 
Core i3 6100 (dual core + hyperthreading @ 3.7ghz) CPU

## Results

- See folder `results/`
- See rainier images all stitched in `results/extra/rainierImageTree.jpg`
- See panoramas from some photos I took from my phone:
- Mtl panorama: source images in `results/extra/mtl`, panoramas in `results/extra/mtlImageTree.jpg`
- Lookout panorama: source images in `results/extra/lookout`, panoramas in `results/extra/lookoutImageTree.jpg`, `results/extra/lookoutNoBlendCroppedScaled.jpg`

## How it works

I followed the assignment description quite closely. 

Image pairs are found by iterating through the image tree. Image trees are built manually 
by looking at the images and predicting which ones match best with eachother. The structure
of this tree defines which image pairs will be used during the stitching algorithm:

### 1. Get Harris keypoints for each image

Uses 3x3 gaussian with sigma = 1, thresholds values that
are lower than 20% of the max corner strength value, non-max suppression by checking
against surrounding 3x3 grid, ignore points near (within 8 pixels) image boundary.

No more than 750 of the 'best' harris keypoints are selected.

### 2. Get orientation for each keypoint

For some rotation invariance, the dominant orientation
for each keypoint is computed by histogram with 10 bins in a 10x10 window, weighted
with a 10x10 gaussian of sigma = 3.5. If necessary, a single keypoint is split into multiple
keypoints. Orientation value of the dominant bin is interpolated by fitting a parabola with
the neighboring bins and getting the vertex

### 3. Compute SIFT-like descriptors for each keypoint

First, the gradient is subtracted from the orientation of
the keypoint from previous step for rotational invariance. SIFT-like, 128 dimension vector
descriptor is created by computing and concatenating orientation histogram in 4x4
sections of 16x16 neighborhood around the keypoint. Magnitudes are scaled by distance
from the keypoint via 16x16 gaussian with sigma = 4. Each histogram is normalized and
thresholded to mitigate the effect of large gradient values.

### 4. Match keypoints for each image pair

‘Distance’ between keypoints is computed by euclidean distance. The ratio
test is used to display the points for which the ratio between the distance of the best
match and the second best match is sufficiently small (threshold = 0.9)

### 5. Compute homography for each image pair

I implemented the RANSAC algorithm pretty plainly. The threshold for being an inlier
is to have a distance of less than 5 pixels. I typically did 500 iterations of the algorithm.

Since the images are put into a tree structure of arbitrary breadth and depth, 
the final homography of each image is computed by recursively matrix-multiplying 
by the homographies of its parent image so that all homographies are computed
with respect to the root node of the tree (the root node has no homography).

### 6. Stitch the final image

As per the assignment description, the bounadries of the final image are calculated by
projecting the corners of all the images of the tree with the inverse homography.
Then the image tree is traversed and each pixel of each image is projected into the 
reference plane of the root image of the tree and its pixel value is added to the 
final image. For pixels in the final image with more than one contribution 
(more than one image succefully projects onto that coordinate), the contributions 
are averaged if the meanBlend flag is enabled. If the meanBlend flag is disabled, 
the first contribution is chosen and the rest of the potential pixel colors are discarded.

### 7. Write final image to a file