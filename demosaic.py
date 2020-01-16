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

# Calculates the sum of the squares of the differences of each channel
# between the final image and the initial image
# TODO: normalize between a value of 0 -> 255
def generateDifferenceImage(final, initial):
  differences = final - initial
  differencesSquared = (differences * differences)
  dB, dG, dR = np.split(differencesSquared, 3, axis=2)
  return (dB + dG + dR).astype("uint8")

def displayImage(name, data):
  cv.imshow(name, data.clip(0, 255).astype("uint8"))

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
owmSeparated = np.zeros((owmHeight, owmWidth, 3), np.float32)
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

# calculate difference image 
differencesSquaredSummed = generateDifferenceImage(owmDemosaiced, oldwell)

differenceImgG = owmGreen - owmRed
differenceImgB = owmBlue - owmRed

# displayImage("differenceImgG", differenceImgG)
# displayImage("differenceImgB", differenceImgB)

differenceImgG = cv.medianBlur(differenceImgG, 3)
differenceImgB = cv.medianBlur(differenceImgB, 3)

# displayImage("differenceImgGb", differenceImgG)
# displayImage("differenceImgBb", differenceImgB)

owmGreen = differenceImgG + owmRed
owmBlue = differenceImgB + owmRed

# merge channels into a single RGB image
owmDemosaicedFreeman = cv.merge([owmBlue, owmGreen, owmRed])

# calculate difference image 
differencesSquaredSummedF = generateDifferenceImage(owmDemosaicedFreeman, oldwell)

# displayImage("differenceImgGBlurred", differenceImgG)

# display the results
# TODO: try to position the images next to each other so user
# doesn't have to move them manually?
displayImage("oldwell", oldwell)
displayImage("owmDemosaiced", owmDemosaiced)
displayImage("owmDemosaicedFreeman", owmDemosaicedFreeman)
displayImage("differencesSquaredSummed", differencesSquaredSummed)
displayImage("differencesSquaredSummedF", differencesSquaredSummedF)

cv.waitKey(0)