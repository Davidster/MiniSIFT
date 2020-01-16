import cv2 as cv
import numpy as np

BLUE = 0
GREEN = 1
RED = 2

"""
This function implements the Bayer pattern 
as described by the assignment requirements:

B R B R . . etc
R G R G . .
B R B R . .
. . . . . .
. . . . . .
e
t
c

"""
def getColorByPosition(i, j):
  if i%2 == 0:
    if j%2 == 0:
      return BLUE
    else:
      return RED
  else:
    if j%2 == 0:
      return RED
    else:
      return GREEN

def applyFilter(image, filter):
  return cv.filter2D(image, -1, filter, borderType=cv.BORDER_ISOLATED)

# Filter used to interpolate blue and green channels
blueGreenFilter = np.array([
  [0.25, 0.5, 0.25], 
  [0.5,  1,   0.5], 
  [0.25, 0.5, 0.25]])

# Filter used to interpolate red channel
redFilter = np.array([
  [0, 0.25, 0], 
  [0.25, 1, 0.25], 
  [0, 0.25, 0]])

# MAIN

# read and parse images from fs
oldwell = cv.imread("image_set/oldwell.jpg")
oldwellMosaic = cv.imread("image_set/oldwell_mosaic.bmp")

# separate mosaic into three color channels
# TODO: convert to list comprehension
owmHeight, owmWidth, d = oldwellMosaic.shape
owmSeparated = np.zeros((owmHeight, owmWidth, 3), np.uint8)
for i, row in enumerate(oldwellMosaic):
  for j, col in enumerate(oldwellMosaic[i]):
    supposedColor = getColorByPosition(i, j)
    owmSeparated[i][j][supposedColor] = col[0]
owmBlue, owmGreen, owmRed = cv.split(owmSeparated)

# run appropriate filter on each channel 
owmBlue = applyFilter(owmBlue, blueGreenFilter)
owmGreen = applyFilter(owmGreen, blueGreenFilter)
owmRed = applyFilter(owmRed, redFilter)

# merge channels into a single RGB image
owmDemosaiced = cv.merge([owmBlue, owmGreen, owmRed])

# calculate root squared 
# TODO: normalize between a value of 0 -> 255
differences = owmDemosaiced - oldwell
differencesSquared = (differences * differences)
dB, dG, dR = np.split(differencesSquared, 3, axis=2)
differencesSquaredSummed = (dB + dG + dR).astype("uint8")

# display the results
# TODO: try to position the images next to each other so user
# doesn't have to move them manually?
cv.imshow("oldwell", oldwell)
cv.imshow("owmDemosaiced", owmDemosaiced)
cv.imshow("differencesSquaredSummed", differencesSquaredSummed)

cv.waitKey(0)