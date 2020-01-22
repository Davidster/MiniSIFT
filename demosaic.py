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
def generateDifferenceImage(final, initial):
  differences = final - initial
  differencesSquared = (differences * differences)
  dB, dG, dR = np.split(differencesSquared, 3, axis=2)
  euclidianDifference = np.sqrt(dB + dG + dR)
  return cv.cvtColor(euclidianDifference.astype("uint8"), cv.COLOR_GRAY2BGR)

def displayImage(name, data):
  cv.imshow(name, data.clip(0, 255).astype("uint8"))

def zoomInOnPencilHead(difference):
  # define rectangle around pencil's head
  closeupRectangle = [[70, 120], [275, 307]]
  # slice image arond rectangle
  closeup = difference[
    closeupRectangle[0][0]:closeupRectangle[0][1], 
    closeupRectangle[1][0]:closeupRectangle[1][1]
  ]
  # zoom in 5x
  closeup = cv.resize(closeup, None, fx=5, fy=5, interpolation=cv.INTER_AREA)
  # draw thin green border
  closeup = cv.copyMakeBorder(closeup, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, (0, 255, 0))
  # draw thick black border such that the height is equal to the original image
  verticalPadding = int((mHeight - closeup.shape[0]) / 2)
  horizontalPadding = 20
  return cv.copyMakeBorder(closeup, verticalPadding, verticalPadding, horizontalPadding, horizontalPadding, cv.BORDER_CONSTANT, None, (0, 0, 0))

# filter used to interpolate blue and green channels
blueGreenFilter = np.array([
  [0.25, 0.5, 0.25], 
  [0.5,  1,   0.5], 
  [0.25, 0.5, 0.25]])

# filter used to interpolate red channel
redFilter = np.array([
  [0, 0.25, 0], 
  [0.25, 1, 0.25], 
  [0, 0.25, 0]])

# PART 1

# read and parse images from fs
realImage = cv.imread("image_set/pencils.jpg")
mosaicedImage = cv.imread("image_set/pencils_mosaic.bmp")
mHeight, mWidth, mChannels = mosaicedImage.shape
print(f'Image shape: {mHeight} x {mWidth} px')

# separate mosaic into three color channels
# TODO: convert to list comprehension
mSeparated = np.zeros((mHeight, mWidth, 3), np.float32)
for i, row in enumerate(mosaicedImage):
  for j, col in enumerate(mosaicedImage[i]):
    mSeparated[i][j][getColorByPosition(i, j)] = col[0]
mBlue, mGreen, mRed = cv.split(mSeparated)

# run appropriate filter on each channel 
mBlue = applyFilter(mBlue, blueGreenFilter)
mGreen = applyFilter(mGreen, blueGreenFilter)
mRed = applyFilter(mRed, redFilter)

# merge channels into a single RGB image
demosaicedImage = cv.merge([mBlue, mGreen, mRed])

# calculate difference image 
demosaicDifferenceImage = generateDifferenceImage(demosaicedImage, realImage)
demosaicDifferenceImageZoomed = zoomInOnPencilHead(demosaicDifferenceImage)

# PART 2 - Freeman tactic

dGR = mGreen - mRed
dBR = mBlue - mRed
dGR = cv.medianBlur(dGR, 5)
dBR = cv.medianBlur(dBR, 5)
mGreen = dGR + mRed
mBlue = dBR + mRed

# merge channels into a single RGB image
freemannedImage = cv.merge([mBlue, mGreen, mRed])

# calculate difference image 
freemanDifferenceImage = generateDifferenceImage(freemannedImage, realImage)
freemanDifferenceImageZoomed = zoomInOnPencilHead(freemanDifferenceImage)

# DISPLAY RESULTS

# arrange the 4 images horiontally into a single image
partOneShowcase = np.concatenate((realImage, demosaicedImage, demosaicDifferenceImage, demosaicDifferenceImageZoomed), axis=1)
displayImage("Part 1 Showcase", partOneShowcase)
partTwoShowcase = np.concatenate((realImage, freemannedImage, freemanDifferenceImage, freemanDifferenceImageZoomed), axis=1)
displayImage("Part 2 Showcase", partTwoShowcase)

cv.waitKey(0)