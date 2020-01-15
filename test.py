import cv2 as cv
import numpy as np

BLUE = 0
GREEN = 1
RED = 2

blueGreenFilter = np.array([
  [0.25, 0.5, 0.25], 
  [0.5,  1,   0.5], 
  [0.25, 0.5, 0.25]], np.float32)
redFilter = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]], np.float32)

oldwell = cv.imread("image_set/oldwell.jpg")
oldwellMosaic = cv.imread("image_set/oldwell_mosaic.bmp")

owmHeight, owmWidth, d = oldwellMosaic.shape
blankChannel = np.ones((owmHeight, owmWidth), np.uint8)

cv.imshow("oldwell", oldwell)
# cv.imshow("oldwellMosaic", oldwellMosaic)

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

owmSeparated = np.zeros((owmHeight, owmWidth, 3), np.uint8)
for i, row in enumerate(oldwellMosaic):
  for j, col in enumerate(oldwellMosaic[i]):
    supposedColor = getColorByPosition(i, j)
    owmSeparated[i][j][supposedColor] = col[0]

# cv.imshow("owmSeparated", owmSeparated)    

owmBlue, owmGreen, owmRed = cv.split(owmSeparated)

owmBlue = cv.filter2D(owmBlue, -1, blueGreenFilter, borderType=cv.BORDER_ISOLATED)
owmGreen = cv.filter2D(owmGreen, -1, blueGreenFilter, borderType=cv.BORDER_ISOLATED)
owmRed = cv.filter2D(owmRed, -1, redFilter, borderType=cv.BORDER_ISOLATED)

# cv.imshow("owmRed", owmRed)

# cv.imshow("owmGreen", owmGreen)

# cv.imshow("owmBlue", owmBlue)

owmDemosaiced = cv.merge([owmBlue, owmGreen, owmRed])

cv.imshow("owmDemosaiced", owmDemosaiced)

oldwellB, oldwellG, oldwellR = cv.split(oldwell)
# cv.imshow("oldwellR", oldwellR)
# cv.imshow("oldwellG", oldwellG)
# cv.imshow("oldwellB", oldwellB)

oldwelldB, oldwelldG, oldwelldR = cv.split(owmDemosaiced)
# cv.imshow("oldwelldR", oldwelldR)
# cv.imshow("oldwelldG", oldwelldG)
# cv.imshow("oldwelldB", oldwelldB)

cv.waitKey(0)