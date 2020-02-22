import cv2 as cv
import numpy as np
from multiprocessing import Pool
import math
import time
import functools
import operator
import random
from PIL import Image, ImageDraw

sobelFilterScale = 1/8
sobelFilterX = np.array([
  [-1,  0,  1], 
  [-2,  0,  2], 
  [-1,  0,  1]]) * sobelFilterScale
sobelFilterY = np.array([
  [ 1,  2,  1], 
  [ 0,  0,  0], 
  [-1, -2, -1]]) * sobelFilterScale

RED_PIXEL = np.array([0, 0, 255]).astype("uint8")
BLUE_PIXEL = np.array([255, 0, 0]).astype("uint8")
GREEN_PIXEL = np.array([0, 255, 0]).astype("uint8")

def applyFilter(image, filter):
  return cv.filter2D(image, -1, filter, borderType=cv.BORDER_ISOLATED)

def getSobelGradient(image):
  return (
    applySobel(image, sobelFilterX).astype("float32"),
    applySobel(image, sobelFilterY).astype("float32"))

def applySobel(image, singleDimensionFilter):
  return cv.cvtColor(
    applyFilter(image, singleDimensionFilter), 
    cv.COLOR_RGB2GRAY)

def getNpGradient(image):
  b, g, r = cv.split(image)
  dxb, dyb = np.gradient(b)
  dxg, dyg = np.gradient(g)
  dxr, dyr = np.gradient(r)
  return (mergeNpGradChannels(dxb, dxg, dxr),
          mergeNpGradChannels(dyb, dyg, dyr))
            
# LOL
def getNpGradient2(image):
  return list(map(lambda dimension: mergeNpGradChannels(*dimension),
            list(zip(*(map(lambda channel: np.gradient(channel), 
                            cv.split(image)))))))

def mergeNpGradChannels(db, dg, dr):
  return cv.cvtColor(
            cv.merge([db, dg, dr]).astype("float32"), 
            cv.COLOR_RGB2GRAY)

def applyGaussian(image, shape = (3, 3), sigma = 1):
  return cv.GaussianBlur(image, shape, sigma, borderType=cv.BORDER_REFLECT_101)

def computeCornerStrengths(aphg, imageshape, threads = 3):
  with Pool(threads) as pool:
    return np.reshape(list(pool.map(
          computeCornerStrength,
          list(map(lambda p: (aphg, p[0], p[1]), np.ndindex(imageshape[0], imageshape[1])))
        )), 
        (imageshape[0], imageshape[1])
      ).astype("float32")

def computeCornerStrength(args):
  aphg, x, y = args
  a = aphg[0][0][x][y]
  b = aphg[0][1][x][y]
  c = aphg[1][0][x][y]
  d = aphg[1][1][x][y]
  return (a * d - b * c) / ((a + d) + 0.00001)

def isMaxAmongNeighbors(image, point):
  pX, pY = point
  if image[pX][pY] == 0:
    return False
  return image[pX, pY] >= np.amax(list(map(
    lambda x: np.amax(list(map(
      lambda y: image[x][y], 
      range(max(pY - 1, 0), min(pY + 2, len(image[x])))))), 
    range(max(pX - 1, 0), min(pX + 2, len(image))))))

def getCircleAroundPoint(pos, radius):
  circ = radius * 2
  shape = (circ + 1, circ + 1)
  image = Image.new("1", shape)
  ImageDraw.Draw(image).ellipse(
    (0, 0, circ, circ), outline = "white")
  return np.array(pilImageToPointArray(image, shape)) + pos - (radius, radius)

def getLineFromPoint(pos, angle, length):
  shape = (length * 2, length * 2)
  center = np.array([length, length])
  image = Image.new("1", shape)
  ImageDraw.Draw(image).line([
    tuple(center),
    tuple(center + np.array([math.cos(angle), math.sin(angle)]) * length)
  ], fill = "white")
  return pilImageToPointArray(image, shape) + pos - (length, length)

def pilImageToPointArray(pilImage, imageShape):
  return np.array(
    list(
      map(
        lambda point: point[0],
        filter(
          lambda point: point[1] == 255,
          np.ndenumerate(np.reshape(pilImage.getdata(), imageShape))
        )
      )
    )
  )

def distance2D(a, b):
  dx = a[0] - b[0]
  dy = a[1] - b[1]
  return math.sqrt(dx * dx + dy * dy)

def computeHarrisKeypoints(image, gradient):
  dx, dy = gradient

  # Build harris matrix
  allPointsHarris = np.array([
    [ dx * dx, dx * dy ],
    [ dy * dx, dy * dy ]
  ]).astype("float32")

  # Apply gaussian weights
  allPointsHarrisWGaussian = np.array(list(
    map(lambda row: list(map(lambda ele: applyGaussian(ele), row)), allPointsHarris)
  ))

  # Compute corner strenths and threshold
  startTime = time.time()
  harrisCornerStrengthImage = computeCornerStrengths(allPointsHarrisWGaussian.astype("float32"), image.shape)
  maxCornerStrength = np.amax(harrisCornerStrengthImage)
  # cv.imshow("nonethresholded", harrisCornerStrengthImage / 100)
  thresholded = (harrisCornerStrengthImage > maxCornerStrength * 0.2) * harrisCornerStrengthImage
  print(f"Corner strength + threshold: {(time.time() - startTime) * 1000}ms")
  # cv.imshow("thresholded", thresholded)

  # Perform non-maximum suppression
  startTime = time.time()
  nonMaxSuppressed = np.reshape(
      np.array(list(map(
        lambda p: isMaxAmongNeighbors(thresholded, p) * thresholded[p[0]][p[1]], 
        np.ndindex(*thresholded.shape)))),
    thresholded.shape)
  keypoints = list(filter(lambda point: point[1] > 0, 
    np.ndenumerate(nonMaxSuppressed)))
  # cv.imshow("nonMaxSuppressed", nonMaxSuppressed)

  keypoints = list(filter(
    lambda point: not pointNearBoundary(point, image.shape),
    keypoints
  ))

  print(f"Non max suppresion: {(time.time() - startTime) * 1000}ms")
  print(f"Found {len(keypoints)} keypoints")

  return keypoints

def annotateKeypoints(image, keypoints):
  annotatedPointMap = {}
  circleRadius = 8
  circleInterleaver = 0
  for keypoint in keypoints:
    pos, val, angle, ringColor = keypoint
    for circlePoint in getCircleAroundPoint(pos, circleRadius):
      circlePoint = tuple(circlePoint) 
      circleInterleaver += 1
      if circlePoint not in annotatedPointMap or circleInterleaver % 2 == 0:
        annotatedPointMap[circlePoint] = ringColor  
    for linePoint in getLineFromPoint(pos, angle, circleRadius):
      linePoint = tuple(linePoint)
      annotatedPointMap[linePoint] = GREEN_PIXEL 
       
  for keypoint in keypoints:
    pos, val, angle, ringColor = keypoint
    annotatedPointMap[pos] = RED_PIXEL  

  annotatedImage = np.copy(image)
  for x, row in enumerate(annotatedImage):
    for y, point in enumerate(row):
      pos = (x, y)
      if pos in annotatedPointMap:
        annotatedImage[x][y] = annotatedPointMap[pos]
  return annotatedImage

# Adapted from https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
def getParabolaVertex(p1, p2, p3):
  x1, y1 = p1
  x2, y2 = p2
  x3, y3 = p3
  denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
  a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
  b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
  c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
  return tuple((-b, (c - b * b) / 2) / (2 * a))

ORIENTATION_WINDOWSIZE = 5
ORIENTATION_GAUSSIAN = cv.getGaussianKernel(ORIENTATION_WINDOWSIZE * 2, 3.5)
def getKeypointOrientation(keypoint, imageShape, gradientMagnitudes, gradientDirections):
  (keypointX, keypointY), val = keypoint
  neighborhood = (
    (max(0, keypointX - ORIENTATION_WINDOWSIZE), max(0, keypointY - ORIENTATION_WINDOWSIZE)), 
    (min(imageShape[0], keypointX + ORIENTATION_WINDOWSIZE), min(imageShape[1], keypointY + ORIENTATION_WINDOWSIZE))
  )
  # print(keypoint)
  # print(neighborhood)
  binCount = 10
  orientationVoteCounts = np.zeros(binCount)
  # print(orientationVoteCounts)
  for x in range(neighborhood[0][0], neighborhood[1][0]):
    for y in range(neighborhood[0][1], neighborhood[1][1]):
      binIndex = math.floor(binCount * gradientDirections[x][y] / (math.pi * 2))
      gaussianFactor = ORIENTATION_GAUSSIAN[x - keypointX] * ORIENTATION_GAUSSIAN[y - keypointY]
      orientationVoteCounts[binIndex] += gaussianFactor * gradientMagnitudes[x][y]
  # print(orientationVoteCounts)

  dominantBinValue = np.amax(orientationVoteCounts)
  peakBinIndices, peakBinValues = map(list, zip(*list(filter(
    lambda vc: vc[1] > dominantBinValue * 0.8,
    list(enumerate(orientationVoteCounts))
  ))))
  # print(peakBinIndices)

  orientedKeypoints = []
  for peakBinIndex in peakBinIndices:
    neighborhood = tuple(map(
      lambda binIndex: (
        (binIndex / binCount) * math.pi * 2, 
        orientationVoteCounts[binIndex % binCount]
      ),
      range(peakBinIndex - 1, peakBinIndex + 2))
    )
    orientedKeypoints.append((*keypoint, getParabolaVertex(*neighborhood)[0]))
  return orientedKeypoints

def normalize(npVector):
  # print(npVector)
  return npVector / (np.sqrt(np.nansum(npVector * npVector)) + 0.0001)

SIFT_WINDOWSIZE = 8
SIFT_SUB_WINDOWSIZE = 4
SIFT_GAUSSIAN = cv.getGaussianKernel(SIFT_WINDOWSIZE * 2, 4)
# print(SIFT_GAUSSIAN)
def getKeypointDescriptor(keypoint, imageShape, gradientMagnitudes, gradientDirections):
  (keypointX, keypointY), val, orientation = keypoint
  neighborhood = (
    (keypointX - SIFT_WINDOWSIZE, keypointY - SIFT_WINDOWSIZE), 
    (keypointX + SIFT_WINDOWSIZE, keypointY + SIFT_WINDOWSIZE)
  )

  # print(keypoint)
  # print(gradientDirections[0:3][0:3])
  gradientDirections = (gradientDirections - keypoint[2] + (math.pi * 2)) % (math.pi * 2)
  # print(gradientDirections[0:3][0:3])
  

  descriptor = []
  for i in range(0, 4):
    for j in range(0, 4):
      startX = neighborhood[0][0] + (i * 4)
      startY = neighborhood[0][1] + (j * 4)
      subNeighborhood = ((startX, startY), (startX + 4, startY + 4))
      # print(f"subNeighborhood: {subNeighborhood}")

      binCount = 8
      orientationVoteCounts = np.zeros(binCount)
      for x in range(subNeighborhood[0][0], subNeighborhood[1][0]):
        for y in range(subNeighborhood[0][1], subNeighborhood[1][1]):
          binIndex = math.floor(binCount * gradientDirections[x][y] / (math.pi * 2))
          gaussianFactor = SIFT_GAUSSIAN[x - keypointX] * SIFT_GAUSSIAN[y - keypointY]
          orientationVoteCounts[binIndex] += gaussianFactor * gradientMagnitudes[x][y]
          
      # if(np.sum(orientationVoteCounts) < 0.001):
      #   print("bruh")
      #   print(keypoint)
      #   print(neighborhood)     
      #   print(orientationVoteCounts)
      #   print(normalize(np.clip(normalize(orientationVoteCounts), None, 0.2)))
      #   print(np.sum(normalize(np.clip(normalize(orientationVoteCounts), None, 0.2))))
        # for x in range(subNeighborhood[0][0], subNeighborhood[1][0]):
        #   for y in range(subNeighborhood[0][1], subNeighborhood[1][1]):
        #     binIndex = math.floor(binCount * gradientDirections[x][y] / (math.pi * 2))
        #     gaussianFactor = SIFT_GAUSSIAN[x - keypointX] * SIFT_GAUSSIAN[y - keypointY]
            # orientationVoteCounts[binIndex] += gaussianFactor * gradientMagnitudes[x][y]
            # print(f"({x}, {y})")
            # print(f"grad: {gradientDirections[x][y]}")
            # print(f"binIndex: {binIndex}")
            # print(f"gaussianFactor: {gaussianFactor}")
            # print(f"gradientMagnitudes[x][y]: {gradientMagnitudes[x][y]}")

      
      # orientationVoteCounts = normalize(np.clip(normalize(orientationVoteCounts), None, 0.2))
      descriptor += list(orientationVoteCounts)

  descriptor = list(normalize(np.clip(normalize(np.array(descriptor)), None, 0.2)))

  # print(f"descriptor length: {len(descriptor)}")
  return (*keypoint, descriptor)

def getKeypointDescriptors(image, gradient, keypoints):
  dx, dy = gradient
  # dx, dy = getNpGradient2(applyGaussian(image))
  gradientMagnitudes = np.sqrt(dx * dx + dy * dy)
  gradientDirections = np.arctan2(dy, dx) + math.pi
  newKeypoints = []
  print("Computing descriptors")
  count = 0
  for keypoint in keypoints:
    # startTime = time.time()
    orientedKeypoints = getKeypointOrientation(keypoint, image.shape, gradientMagnitudes, gradientDirections)
    descriptedKeypoints = list(map(
      lambda kp: getKeypointDescriptor(kp, image.shape, gradientMagnitudes, gradientDirections),
      orientedKeypoints
    ))
    for n in range(0, len(orientedKeypoints)):
      count += 1
      if(count % 100 == 0):
        print(f"Done {count} / {len(keypoints)} keypoints")
    # print(f"Keypoint done in: {(time.time() - startTime) * 1000}ms")
    # print(f"peakBinIndex: {peakBinIndex}")
    # print(f"neighborhood: {neighborhood}")
    # print(f"parabola vertex: {getParabolaVertex(*neighborhood)}")
    # print((*keypoint, getParabolaVertex(*neighborhood)[0]))
    newKeypoints = newKeypoints + descriptedKeypoints
    # newKeypoints.append((*keypoint, neighborhood[1][0]))


  # print(newKeypoints[0:5])
  print(f"Done {count} / {len(keypoints)} keypoints")
  print(f"New keypoint count: {len(newKeypoints)}")
  return newKeypoints

def getBestMatchesByThreshold(img1Keypoints, img2Keypoints):
  matches = []
  startTime = time.time()
  count = 0
  for img1Keypoint in img1Keypoints:
    for img2Keypoint in img2Keypoints:
      diff = np.array(img2Keypoint[3]) - np.array(img1Keypoint[3])
      matches.append((img1Keypoint[:3], img2Keypoint[:3], np.sqrt(np.sum(diff * diff))))
      count += 1
      if count % 20000 == 0:
        print(f"Computed {count} / {len(img1Keypoints) * len(img2Keypoints)} keypoint distances")
  print(f"Matches computed in {(time.time() - startTime) * 1000}ms")
  
  startTime = time.time()
  sortedMatches = sorted(matches, key=lambda pair: pair[2])

  maxDistance = 0.75
  filteredMatches = list(filter(lambda match: match[2] < maxDistance, sortedMatches))

  all1s = set()
  all2s = set()
  for match in filteredMatches:
    all1s.add(match[0][0]) 
    all2s.add(match[1][0])

  used1s = set()
  used2s = set()
  topMatches = []
  while True:
    foundMatch = False
    for match in filteredMatches:
      match1, match2, distance = match
      match1 = match1[0]
      match2 = match2[0]
      if match1 in used1s or match2 in used2s:
        continue
      foundMatch = True
      used1s.add(match1) 
      used2s.add(match2)
      topMatches.append(match)
      break
    if not foundMatch or len(used1s) == len(all1s) or len(used2s) == len(all2s):
      print("No more unique matches to find")
      break

  print(f"Matches ranked in {(time.time() - startTime) * 1000}ms")
  matchedRatio = len(topMatches) / len(sortedMatches)
  print(f"Selectd {len(topMatches)} (top {round(matchedRatio, 5) * 100}%) matches")
  print(f"Match distance range: {topMatches[0][2]} : {topMatches[-1][2]}")

  img1MatchedKeypoints = []
  img2MatchedKeypoints = []
  for match in topMatches:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img1MatchedKeypoints.append((*match[0], color))
    img2MatchedKeypoints.append((*match[1], color))

  return (img1MatchedKeypoints, img2MatchedKeypoints)

def getBestMatchesByRatioTest(img1Keypoints, img2Keypoints):
  matches = []
  startTime = time.time()
  count = 0
  for img1Keypoint in img1Keypoints:
    for img2Keypoint in img2Keypoints:
      diff = np.array(img2Keypoint[3]) - np.array(img1Keypoint[3])
      matches.append((img1Keypoint[:3], img2Keypoint[:3], np.sqrt(np.sum(diff * diff))))
      count += 1
      if count % 20000 == 0:
        print(f"Computed {count} / {len(img1Keypoints) * len(img2Keypoints)} keypoint distances")
  print(f"Matches computed in {(time.time() - startTime) * 1000}ms")
  
  startTime = time.time()
  sortedMatches = sorted(matches, key=lambda pair: pair[2])

  count = 0
  ratioMatches = []
  for keypoint in img1Keypoints:
    bestMatch, secondBestMatch = list(filter(lambda match: match[0][0] == keypoint[0], sortedMatches))[:2]
    # print(f"keypoint: {keypoint[0]}\nbestMatch: {bestMatch}\nsecondBestMatch: {secondBestMatch}\nratio:{bestMatch[2]/secondBestMatch[2]}")
    ratioMatches.append((bestMatch[0], bestMatch[1], bestMatch[2]/secondBestMatch[2]))
    count += 1
    if count % 25 == 0:
      print(f"Computed {count} / {len(img1Keypoints) + len(img2Keypoints)} keypoint ratios")
  for keypoint in img2Keypoints:
    bestMatch, secondBestMatch = list(filter(lambda match: match[1][0] == keypoint[0], sortedMatches))[:2]
    ratioMatches.append((bestMatch[0], bestMatch[1], bestMatch[2]/secondBestMatch[2]))
    count += 1
    if count % 25 == 0:
      print(f"Computed {count} / {len(img1Keypoints) + len(img2Keypoints)} keypoint ratios")

  sortedRatioMatches = sorted(ratioMatches, key=lambda pair: pair[2])

  maxDistance = 0.85
  filteredMatches = list(filter(lambda match: match[2] < maxDistance, sortedRatioMatches))

  all1s = set()
  all2s = set()
  for match in filteredMatches:
    all1s.add(match[0][0]) 
    all2s.add(match[1][0])

  used1s = set()
  used2s = set()
  topMatches = []
  while True:
    foundMatch = False
    for match in filteredMatches:
      match1, match2, distance = match
      match1 = match1[0]
      match2 = match2[0]
      if match1 in used1s or match2 in used2s:
        continue
      foundMatch = True
      used1s.add(match1) 
      used2s.add(match2)
      topMatches.append(match)
      break
    if not foundMatch or len(used1s) == len(all1s) or len(used2s) == len(all2s):
      print("No more unique matches to find")
      break

  print(f"Matches ranked in {(time.time() - startTime) * 1000}ms")
  matchedRatio = len(topMatches) / len(sortedMatches)
  print(f"Selectd {len(topMatches)} (top {round(matchedRatio, 5) * 100}%) matches")
  print(f"Match distance range: {topMatches[0][2]} : {topMatches[-1][2]}")

  img1MatchedKeypoints = []
  img2MatchedKeypoints = []
  for match in topMatches:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img1MatchedKeypoints.append((*match[0], color))
    img2MatchedKeypoints.append((*match[1], color))

  return (img1MatchedKeypoints, img2MatchedKeypoints)

def pointNearBoundary(point, imageShape):
  (x, y), val = point
  return (
    x < SIFT_WINDOWSIZE or
    y < SIFT_WINDOWSIZE or
    x > imageShape[0] - SIFT_WINDOWSIZE or
    y > imageShape[1] - SIFT_WINDOWSIZE
  )

if __name__== "__main__":
  # img1 = cv.imread("image_sets/graf/img1.ppm")
  # img2 = cv.imread("image_sets/graf/img2.ppm")
  # img1 = cv.imread("image_sets/panorama/pano1_0009.png")
  # img2 = cv.imread("image_sets/panorama/pano1_0008.png")
  img1 = cv.imread("image_sets/yosemite/Yosemite1.jpg")
  img2 = cv.imread("image_sets/yosemite/Yosemite2.jpg")

  print("-- Harris Keypoints --") 
  print("-- Image 1 --")
  img1Gradient = getNpGradient2(img1)
  img1HarrisKeypoints = computeHarrisKeypoints(img1, img1Gradient)
  print("-- Image 2 --")
  img2Gradient = getNpGradient2(img2)
  img2HarrisKeypoints = computeHarrisKeypoints(img2, img2Gradient)

  print("-- SIFT Descriptors --")
  print("-- Image 1 --")
  img1DescriptedKeypoints = getKeypointDescriptors(img1, img1Gradient, img1HarrisKeypoints)
  print("-- Image 2 --")
  img2DescriptedKeypoints = getKeypointDescriptors(img2, img2Gradient, img2HarrisKeypoints)

  print("-- SSD Matches --")
  img1SSDKeypoints, img2SSDKeypoints = getBestMatchesByThreshold(img1DescriptedKeypoints, img2DescriptedKeypoints)
  cv.imshow("img1 SSD", annotateKeypoints(img1, img1SSDKeypoints))
  cv.imshow("img2 SSD", annotateKeypoints(img2, img2SSDKeypoints))

  print("-- Ratio Test Matches --")
  img1RatioKeypoints, img2RatioKeypoints = getBestMatchesByRatioTest(img1DescriptedKeypoints, img2DescriptedKeypoints)
  cv.imshow("img1 ratio", annotateKeypoints(img1, img1RatioKeypoints))
  cv.imshow("img2 ratio", annotateKeypoints(img2, img2RatioKeypoints))

  cv.waitKey(0)
  # grafImg4Gradient = getSobelGradient(grafImg4)
  # grafImg4Keypoints = computeHarrisKeypoints(grafImg4, grafImg4Gradient)

  
  # cv.imshow("grafImg4", annotateKeypoints(grafImg4, grafImg4Keypoints))
  # cv.imshow("original", image)
  # cv.imshow("harrisCornerStrengthImage", harrisCornerStrengthImage)
  # cv.imshow("thresholded", thresholded)
  # cv.imshow("nonMaxSuppressed", nonMaxSuppressed)
  # cv.imshow("annotatedImage", annotatedImage)
  