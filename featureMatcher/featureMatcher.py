import cv2 as cv
import numpy as np
from multiprocessing import Pool
import math
import time
import functools
import operator
import random
import itertools
import os
from PIL import Image, ImageDraw

# sobelFilterScale = 1/8
# sobelFilterX = np.array([
#   [-1,  0,  1], 
#   [-2,  0,  2], 
#   [-1,  0,  1]]) * sobelFilterScale
# sobelFilterY = np.array([
#   [ 1,  2,  1], 
#   [ 0,  0,  0], 
#   [-1, -2, -1]]) * sobelFilterScale

RED_PIXEL = np.array([0, 0, 255]).astype("uint8")
BLUE_PIXEL = np.array([255, 0, 0]).astype("uint8")
GREEN_PIXEL = np.array([0, 255, 0]).astype("uint8")

THREADS = 3
try:
  THREADS = os.cpu_count() * 2
except:
  print(f"Could not get os.cpu_count(). Falling back on {THREADS} 'threads'.")

# def applyFilter(image, filter):
#   return cv.filter2D(image, -1, filter, borderType=cv.BORDER_ISOLATED)

# def getSobelGradient(image):
#   return (
#     applySobel(image, sobelFilterX).astype("float32"),
#     applySobel(image, sobelFilterY).astype("float32"))

# def applySobel(image, singleDimensionFilter):
#   return cv.cvtColor(
#     applyFilter(image, singleDimensionFilter), 
#     cv.COLOR_RGB2GRAY)

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

def computeCornerStrengths(aphg, imageshape):
  with Pool(THREADS) as pool:
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

MAX_KEYPOINTS_PER_IMAGE = 750
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
  keypoints = filter(lambda point: point[1] > 0, 
    np.ndenumerate(nonMaxSuppressed))
  # cv.imshow("nonMaxSuppressed", nonMaxSuppressed)

  keypoints = filter(
    lambda point: not pointNearBoundary(point, image.shape),
    keypoints
  )

  keypoints = sorted(keypoints, 
    key=lambda keypoint: keypoint[1], reverse=True)

  keypoints = list(itertools.islice(keypoints, MAX_KEYPOINTS_PER_IMAGE))

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

  gradientDirections = (gradientDirections + (math.pi * 2)) % (math.pi * 2)

  binCount = 10
  orientationVoteCounts = np.zeros(binCount)
  for x in range(neighborhood[0][0], neighborhood[1][0]):
    for y in range(neighborhood[0][1], neighborhood[1][1]):
      binIndex = math.floor(binCount * gradientDirections[x][y] / (math.pi * 2))
      gaussianFactor = ORIENTATION_GAUSSIAN[x - keypointX] * ORIENTATION_GAUSSIAN[y - keypointY]
      orientationVoteCounts[binIndex] += gaussianFactor * gradientMagnitudes[x][y]

  dominantBinValue = np.amax(orientationVoteCounts)
  peakBinIndices, peakBinValues = map(list, zip(*list(filter(
    lambda vc: vc[1] > dominantBinValue * 0.8,
    list(enumerate(orientationVoteCounts))
  ))))

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
  return npVector / (np.sqrt(np.nansum(npVector * npVector)) + 0.0001)

SIFT_WINDOWSIZE = 8
SIFT_SUB_WINDOWSIZE = 4
SIFT_GAUSSIAN = cv.getGaussianKernel(SIFT_WINDOWSIZE * 2, 4)
def getKeypointDescriptor(keypoint, imageShape, gradientMagnitudes, gradientDirections):
  (keypointX, keypointY), val, orientation = keypoint
  neighborhood = (
    (keypointX - SIFT_WINDOWSIZE, keypointY - SIFT_WINDOWSIZE), 
    (keypointX + SIFT_WINDOWSIZE, keypointY + SIFT_WINDOWSIZE)
  )

  gradientDirections = (gradientDirections - keypoint[2] + (math.pi * 2)) % (math.pi * 2)

  descriptor = []
  for i in range(0, 4):
    for j in range(0, 4):
      startX = neighborhood[0][0] + (i * 4)
      startY = neighborhood[0][1] + (j * 4)
      subNeighborhood = ((startX, startY), (startX + 4, startY + 4))

      binCount = 8
      orientationVoteCounts = np.zeros(binCount)
      for x in range(subNeighborhood[0][0], subNeighborhood[1][0]):
        for y in range(subNeighborhood[0][1], subNeighborhood[1][1]):
          binIndex = math.floor(binCount * gradientDirections[x][y] / (math.pi * 2))
          gaussianFactor = SIFT_GAUSSIAN[x - keypointX] * SIFT_GAUSSIAN[y - keypointY]
          orientationVoteCounts[binIndex] += gaussianFactor * gradientMagnitudes[x][y]

      descriptor += list(orientationVoteCounts)

  descriptor = list(normalize(np.clip(normalize(np.array(descriptor)), None, 0.2)))
  return (*keypoint, descriptor)

def getKeypointDescriptors(image, gradient, keypoints):
  dx, dy = gradient
  startTime = time.time()
  gradientMagnitudes = np.sqrt(dx * dx + dy * dy)
  gradientDirections = np.arctan2(dy, dx) + math.pi
  newKeypoints = []
  count = 0
  for keypoint in keypoints:
    orientedKeypoints = getKeypointOrientation(keypoint, image.shape, gradientMagnitudes, gradientDirections)
    descriptedKeypoints = list(map(
      lambda kp: getKeypointDescriptor(kp, image.shape, gradientMagnitudes, gradientDirections),
      orientedKeypoints
    ))
    count += 1
    if(count % 100 == 0):
      print(f"Done {count} / {len(keypoints)} keypoints")
    newKeypoints = newKeypoints + descriptedKeypoints

  print(f"Computed {len(keypoints)} descriptors in {(time.time() - startTime) * 1000}ms")
  print(f"New keypoint count: {len(newKeypoints)}")
  return newKeypoints

def computePointDistance(a, b):
  diff = np.array(list(a)) - np.array(list(b))
  return np.sqrt(np.sum(diff * diff))

def computeDistance(args):
  keypoint1, keypoint2 = args
  diff = np.array(keypoint2[3]) - np.array(keypoint1[3])
  return (keypoint1[:3], keypoint2[:3], np.sqrt(np.sum(diff * diff)))

WORKER_CHUNK_SIZE=100000
def getSortedKeypointPairs(img1Keypoints, img2Keypoints):
  allPairs = []
  startTime = time.time()
  with Pool(THREADS) as pool:
    distanceMappings = pool.imap_unordered(
      computeDistance,
      itertools.product(img1Keypoints, img2Keypoints),
      WORKER_CHUNK_SIZE
    )
    for i, result in enumerate(distanceMappings, 1):
      if i % WORKER_CHUNK_SIZE == 0:
        print(f"Computed {i} / {len(img1Keypoints) * len(img2Keypoints)} keypoint distances")
      allPairs.append(result)
  print(f"Computed {len(img1Keypoints) * len(img2Keypoints)} keypoint distances in {(time.time() - startTime) * 1000}ms")

  startTime = time.time()
  sortedPairs = sorted(allPairs, key=lambda pair: pair[2])
  print(f"Keypoint distances sorted in {(time.time() - startTime) * 1000}ms")
  return sortedPairs

def getBestMatchesByThreshold(sortedKeypointPairs):
  startTime = time.time()
  maxDistance = 0.7
  topMatches = list(filter(lambda match: match[2] < maxDistance, sortedKeypointPairs))
  topMatches = dedupePoints(topMatches)

  print(f"Matches filtered in {(time.time() - startTime) * 1000}ms")
  matchedRatio = len(topMatches) / len(sortedKeypointPairs)
  print(f"Selectd {len(topMatches)} (top {round(matchedRatio, 5) * 100}%) matches")
  print(f"Match distance range: {topMatches[0][2]} : {topMatches[-1][2]}")

  return colorizeMatches(topMatches)

def getBestMatchesByRatioTest(img1Keypoints, img2Keypoints, sortedKeypointPairs):
  startTime = time.time()
  count = 0
  ratioMatches = []
  for keypoint in img1Keypoints:
    bestMatch, secondBestMatch = itertools.islice(filter(lambda match: match[0][0] == keypoint[0], sortedKeypointPairs), 2)
    # print(f"keypoint: {keypoint[0]}\nbestMatch: {bestMatch}\nsecondBestMatch: {secondBestMatch}\nratio:{bestMatch[2]/secondBestMatch[2]}")
    ratioMatches.append((bestMatch[0], bestMatch[1], bestMatch[2]/secondBestMatch[2]))
    count += 1
    if count % 250 == 0:
      print(f"Computed {count} / {len(img1Keypoints) + len(img2Keypoints)} keypoint ratios")
  for keypoint in img2Keypoints:
    bestMatch, secondBestMatch = itertools.islice(filter(lambda match: match[1][0] == keypoint[0], sortedKeypointPairs), 2)
    ratioMatches.append((bestMatch[0], bestMatch[1], bestMatch[2]/secondBestMatch[2]))
    count += 1
    if count % 250 == 0:
      print(f"Computed {count} / {len(img1Keypoints) + len(img2Keypoints)} keypoint ratios")
  print(f"Computed {len(img1Keypoints) + len(img2Keypoints)} keypoint ratios in {(time.time() - startTime) * 1000}ms")

  startTime = time.time()
  sortedRatioMatches = sorted(ratioMatches, key=lambda pair: pair[2])
  # maxDistance = 0.95 # for myRoomRotated
  maxDistance = 0.85
  topMatches = list(filter(lambda match: match[2] < maxDistance, sortedRatioMatches))
  topMatches = dedupePoints(topMatches)

  print(f"Matches ranked in {(time.time() - startTime) * 1000}ms")
  matchedRatio = len(topMatches) / len(sortedKeypointPairs)
  print(f"Selectd {len(topMatches)} (top {round(matchedRatio, 5) * 100}%) matches")
  print(f"Match distance range: {topMatches[0][2]} : {topMatches[-1][2]}")

  return colorizeMatches(topMatches)

def dedupePoints(keypointPairs):
  all1s = set()
  all2s = set()
  for pair in keypointPairs:
    all1s.add(pair[0][0]) 
    all2s.add(pair[1][0])

  used1s = set()
  used2s = set()
  deduped = []
  while True:
    foundMatch = False
    for pair in keypointPairs:
      pImg1, pImg2, distance = pair
      pImg1 = pImg1[0]
      pImg2 = pImg2[0]
      if pImg1 in used1s or pImg2 in used2s:
        continue
      foundMatch = True
      used1s.add(pImg1) 
      used2s.add(pImg2)
      deduped.append(pair)
      break
    if not foundMatch or len(used1s) == len(all1s) or len(used2s) == len(all2s):
      print("No more unique matches to find")
      break
  return deduped

def colorizeMatches(matches):
  img1Matches = []
  img2Matches = []
  for match in matches:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img1Matches.append((*match[0], color))
    img2Matches.append((*match[1], color))
  
  return (img1Matches, img2Matches)

def pointNearBoundary(point, imageShape):
  (x, y), val = point
  return (
    x < SIFT_WINDOWSIZE or
    y < SIFT_WINDOWSIZE or
    x > imageShape[0] - SIFT_WINDOWSIZE or
    y > imageShape[1] - SIFT_WINDOWSIZE
  )

def project(point, h):
  if h is None:
    return point
  x, y = point
  denominator = (h[2][0] * x + h[2][1] * y + h[2][2]) + 0.00001
  return (
    (h[0][0] * x + h[0][1] * y + h[0][2]) / denominator,
    (h[1][0] * x + h[1][1] * y + h[1][2]) / denominator
  )

INLIER_THRESHOLD = 5
def computeInliers(img1Keypoints, img2Keypoints, h):
  inlierIndices = []
  for i, img1Keypoint in enumerate(img1Keypoints):
    distance = computePointDistance(project(img1Keypoint[0], h), img2Keypoints[i][0])
    if (distance < INLIER_THRESHOLD):
      inlierIndices.append(i)
  return inlierIndices

def convertToKeypointType(keypoints):
  return list(map(lambda keypoint: cv.KeyPoint(keypoint[0][1], keypoint[0][0], 1), keypoints))

def convertToPointType(keypoints):
  return list(map(lambda keypoint: keypoint[0], keypoints))

def getDrawMatchesImg(img1, img1Keypoints, img2, img2Keypoints):
  return cv.drawMatches(
    img1, 
    convertToKeypointType(img1Keypoints), 
    img2, 
    convertToKeypointType(img2Keypoints), 
    list(map(lambda i: cv.DMatch(i, i, 1), range(len(img1Keypoints)))),
     None
  )

def getHomographyFromKeypoints(img1Keypoints, img2Keypoints):
  img1Points, img2Points = np.float32((convertToPointType(img1Keypoints), convertToPointType(img2Keypoints)))
  h, _ = cv.findHomography(img1Points, img2Points, 0)
  return h

def doRANSAC(img1Keypoints, img2Keypoints, numIterations):
  startTime = time.time()

  pointCount = len(img1Keypoints)
  numberOfPossibleQuadruples = int(math.factorial(pointCount) / (math.factorial(pointCount - 4) * math.factorial(4)))
  numIterations = min(numIterations, numberOfPossibleQuadruples)
  chosenIndexQuadruples = set()
  results = []
  for _ in range(numIterations):
    
    randomIndices = None
    while True:
      randomIndices = random.sample(range(0, len(img1Keypoints)), 4)
      if not tuple(randomIndices) in chosenIndexQuadruples:
        chosenIndexQuadruples.add(tuple(randomIndices))
        break
    
    h = getHomographyFromKeypoints(
      list(map(lambda i: img1Keypoints[i], randomIndices)), 
      list(map(lambda i: img2Keypoints[i], randomIndices))
    )
    inliers = computeInliers(img1Keypoints, img2Keypoints, h)
    results.append((h, inliers))

  resultsSorted = sorted(results, key=lambda result: len(result[1]), reverse=True)
  
  print(f"Done RANSAC in {(time.time() - startTime) * 1000}ms")
  print(f"Best quadruplet found produced {len(resultsSorted[0][1])} inliers")

  finalImg1Inliers = list(map(lambda i: img1Keypoints[i], resultsSorted[0][1]))
  finalImg2Inliers = list(map(lambda i: img2Keypoints[i], resultsSorted[0][1]))

  finalHomography = getHomographyFromKeypoints(finalImg1Inliers, finalImg2Inliers)

  return (
    finalHomography,
    finalImg1Inliers,
    finalImg2Inliers
  )

def doSIFT(imgNode):
  imgKeypoints = None
  if "keypoints" in imgNode:
    imgKeypoints = imgNode["keypoints"]

  if imgKeypoints is None:
    print("-- Harris keypoints --")
    imgGradient = getNpGradient2(imgNode["img"])
    imgHarrisKeypoints = computeHarrisKeypoints(imgNode["img"], imgGradient)
    print("-- SIFT Descriptors --")
    imgKeypoints = getKeypointDescriptors(imgNode["img"], imgGradient, imgHarrisKeypoints)
    imgNode["keypoints"] = imgKeypoints
  else:
    print("Keypoints already computed. Skipping this step")

def getHomographyFromImages(img1Node, img2Node):
  print("-- Image 1 --")
  doSIFT(img1Node)
  print("-- Image 2 --")
  doSIFT(img2Node)

  print("-- Matching keypoints --")
  sortedKeypointPairs = getSortedKeypointPairs(img1Node["keypoints"], img2Node["keypoints"])

  # print("-- SSD Matches --")
  # img1MatchedKeypoints, img2MatchedKeypoints = getBestMatchesByThreshold(sortedKeypointPairs)

  print("-- Ratio Test Matches --")
  img1MatchedKeypoints, img2MatchedKeypoints = getBestMatchesByRatioTest(img1Node["keypoints"], img2Node["keypoints"], sortedKeypointPairs)

  # cv.imshow("img1 ratio", annotateKeypoints(img1, img1RatioKeypoints))
  # cv.imshow("img2 ratio", annotateKeypoints(img2, img2RatioKeypoints))
  # cv.imshow("img1 SSD", annotateKeypoints(img1, img1SSDKeypoints))
  # cv.imshow("img2 SSD", annotateKeypoints(img2, img2SSDKeypoints))
  # cv.imshow("img1 ratio", annotateKeypoints(img1, img1RatioKeypoints))
  # cv.imshow("img2 ratio", annotateKeypoints(img2, img2RatioKeypoints))
  
  finalHomography, finalImg1Inliers, finalImg2Inliers = doRANSAC(img1MatchedKeypoints, img2MatchedKeypoints, 500)
  # cv.imshow("drawnMatches", getDrawMatchesImg(img1Node["img"], img1MatchedKeypoints, img2Node["img"], img2MatchedKeypoints))
  # cv.imshow("drawnMatches inliers", getDrawMatchesImg(img1Node["img"], finalImg1Inliers, img2Node["img"], finalImg2Inliers))

  return finalHomography

def getImageCorners(img):
  return [
    (0, 0),            (img.shape[0], 0), 
    (0, img.shape[1]), (img.shape[0], img.shape[1]) 
  ]

def getProjectedImageCorners(img, h):
  if h is None:
    return getImageCorners(img)
  hInv = np.linalg.inv(h)
  return list(map(lambda corner: project(corner, hInv), getImageCorners(img)))

def computeImageTreeHomographies(node, parentNode = None):
  if parentNode is None:
    node["h"] = None
  else:
    h = getHomographyFromImages(parentNode, node)
    if not parentNode["h"] is None:
      node["h"] = np.matmul(h, parentNode["h"])
    else:
      node["h"] = h
  if "children" in node:
    for child in node["children"]:
      computeImageTreeHomographies(child, parentNode = node)

# imageList is output reference
def flattenImageTree(node, imageList):
  imageList.append(node)
  if "children" in node:
    for child in node["children"]:
      flattenImageTree(child, imageList)

def tryToProjectNode(point, node):
  projX, projY = project(point, node["h"])
  if projX > 0 and projX < node["img"].shape[0] and projY > 0 and projY < node["img"].shape[1]:
    return cv.getRectSubPix(node["img"], (1, 1), (projY, projX))[0][0]

imageNodeList = []
def doMeanBlend(point):
  successfulProjections = []
  for node in imageNodeList:
    projection = tryToProjectNode(point, node)
    if not projection is None:
      successfulProjections.append(projection)
  if len(successfulProjections) == 1:
    return (point, successfulProjections[0])
  elif len(successfulProjections) > 1:
    return (point, np.mean(np.array(successfulProjections), axis=0))

def doFirstSuccessBlend(point):
  successfulProjections = []
  for node in imageNodeList:
    projection = tryToProjectNode(point, node)
    if not projection is None:
      return (point, projection)
  
def stitchImageTree(imageTree, meanBlend = True):
  blendFunction = doMeanBlend
  if not meanBlend:
    blendFunction = doFirstSuccessBlend

  global imageNodeList
  flattenImageTree(imageTree, imageNodeList)

  allCorners = []
  projectedImageCorners = list(map(lambda node: getProjectedImageCorners(node["img"], node["h"]), imageNodeList))
  for cornerList in projectedImageCorners:
    allCorners = allCorners + cornerList
  xValues = list(map(lambda corner: corner[0], allCorners))
  yValues = list(map(lambda corner: corner[1], allCorners))
  xMin = math.floor(np.amin(xValues))
  xMax = math.ceil(np.amax(xValues))
  yMin = math.floor(np.amin(yValues))
  yMax = math.ceil(np.amax(yValues))
  newImage = np.zeros((xMax - xMin, yMax - yMin, 3), np.uint8)
  totalPixelCount = newImage.shape[0] * newImage.shape[1]
  print(f"Panorama shape: {newImage.shape}")

  startTime = time.time()
  # Runs at about 50 to 100 pixels/ms on Core i3 6100 (dual core + hyperthreading @ 3.7ghz)
  with Pool(THREADS) as pool:
    blendedPoints = pool.imap_unordered(
      blendFunction,
      map(
        lambda point: (point[0] + xMin, point[1] + yMin),
        itertools.product(range(newImage.shape[0]), range(newImage.shape[1]))),
      WORKER_CHUNK_SIZE
    )
    for i, result in enumerate(blendedPoints):
      if not i == 0 and i % WORKER_CHUNK_SIZE == 0:
        secondsRemaining = (totalPixelCount - i) / (i / (time.time() - startTime)) # (remaining pixels) / (rate)
        print(f"Done stiching {i} / {totalPixelCount} pixels (ETA: ~{secondsRemaining} seconds)")
      if not result is None:
        point, value = result
        newImage[point[0] - xMin][point[1] - yMin] = value

  print(f"Done stiching {totalPixelCount} pixels in {(time.time() - startTime) * 1000}ms")
  
  return newImage

def buildScaleFixer(img1, img2):
  s = img1.shape[0] / img2.shape[0]
  return np.array([
    [ 1,   1,   s ],
    [ 1,   1,   s ],
    [ 1/s, 1/s, 1 ]
  ])
    
if __name__== "__main__":
  # img1 = cv.imread("image_sets/yosemite/Yosemite1.jpg")
  # img2 = cv.imread("image_sets/yosemite/Yosemite2.jpg")
  # img1 = cv.imread("image_sets/myroom/left.jpg")
  # img2 = cv.imread("image_sets/myroom/right.jpg")
  # img1 = cv.imread("image_sets/myroom/straight.jpg")
  # img2 = cv.imread("image_sets/myroom/rotated.jpg")
  # img1 = cv.imread("image_sets/mtl/left.jpg")
  # img2 = cv.imread("image_sets/mtl/right.jpg")
  # img1 = cv.imread("image_sets/project_images/ND1.png")
  # img2 = cv.imread("image_sets/project_images/ND2.png")
  myRoomImageTree = {
    "img": cv.imread("image_sets/myroom/left.jpg"),
    "children": [
      { "img":cv.imread("image_sets/myroom/right.jpg") }
    ]
  }
  myRoomRotatedImageTree = {
    "img": cv.imread("image_sets/myroom/straight.jpg"),
    "children": [
      { "img":cv.imread("image_sets/myroom/rotated.jpg") }
    ]
  }
  hangingImageTree = {
    "img": cv.imread("image_sets/project_images/Hanging1.png"),
    "children": [
      { "img":cv.imread("image_sets/project_images/Hanging2.png") }
    ]
  }
  grafImageTree = {
    "img": cv.imread("image_sets/graf/img1.ppm"),
    "children": [
      { "img": cv.imread("image_sets/graf/img2.ppm") }
    ]
  }
  panoImageTree = {
    "img": cv.imread("image_sets/panorama/pano1_0009.png"),
    "children": [
      { "img": cv.imread("image_sets/panorama/pano1_0008.png") }
    ]
  }
  panoImageTree2X = {
    "img": cv.imread("image_sets/panorama/pano1_0009_2x.png"),
    "children": [
      { "img": cv.imread("image_sets/panorama/pano1_0008_2x.png") }
    ]
  }
  # Rainier1.png -> Rainier2.png homography: [[ 1.14430474e+00,  1.41475039e-01, -2.88250204e+01], [-4.96256056e-02,  1.24277991e+00, -1.93345058e+02], [-5.86168103e-05,  5.03821231e-04,  1.00000000e+00]]
  rainerImageTree1 = {
    "img": cv.imread("image_sets/project_images/Rainier1.png"),
    "children": [
      { "img": cv.imread("image_sets/project_images/Rainier2.png") }
    ]
  }
  rainerImageTree2 = {
    "img": cv.imread("image_sets/project_images/Rainier1.png"),
    "children": [
      { "img": cv.imread("image_sets/project_images/Rainier2.png") },
      { "img": cv.imread("image_sets/project_images/Rainier3.png") }
    ]
  }
  rainerImageTree3 = {
    "img": cv.imread("image_sets/project_images/Rainier1.png"),
    "children": [
      { "img": cv.imread("image_sets/project_images/Rainier2.png") },
      { "img": cv.imread("image_sets/project_images/Rainier3.png") },
      { 
        "img": cv.imread("image_sets/project_images/Rainier5.png"),
        "children": [
          { "img": cv.imread("image_sets/project_images/Rainier4.png") },
          { "img": cv.imread("image_sets/project_images/Rainier6.png") }
        ] 
      }
    ]
  }
  lookoutImageTree = {
    "img": cv.imread("image_sets/lookout/2.jpg"),
    "children": [
      { "img": cv.imread("image_sets/lookout/1.jpg") },
      {
        "img": cv.imread("image_sets/lookout/3.jpg"),
        "children": [ 
          { "img": cv.imread("image_sets/lookout/4.jpg") } 
        ]
      }
    ]
  }
  lookoutImageTreeHD = {
    "img": cv.imread("image_sets/lookout/2_full.jpg"),
    "children": [
      { "img": cv.imread("image_sets/lookout/1_full.jpg") },
      {
        "img": cv.imread("image_sets/lookout/3_full.jpg"),
        "children": [ 
          { "img": cv.imread("image_sets/lookout/4_full.jpg") } 
        ]
      }
    ]
  }

  startTime = time.time()
  
  imageTree = rainerImageTree3
  computeImageTreeHomographies(imageTree)
  panorama = stitchImageTree(imageTree)

  # for panorama 2x:
  # computeImageTreeHomographies(panoImageTree)
  # panoImageTree2X["h"] = None
  # panoImageTree2X["children"][0]["h"] = panoImageTree["children"][0]["h"] * buildScaleFixer(panoImageTree2X["children"][0]["img"], panoImageTree["children"][0]["img"])
  # panorama = stitchImageTree(panoImageTree2X, meanBlend = False)

  # for HD lookout:
  # computeImageTreeHomographies(lookoutImageTree)
  # scaleFixer = buildScaleFixer(lookoutImageTreeHD["img"], lookoutImageTree["img"])
  # lookoutImageTreeHD["h"] = None
  # lookoutImageTreeHD["children"][0]["h"] = lookoutImageTree["children"][0]["h"] * scaleFixer
  # lookoutImageTreeHD["children"][1]["h"] = lookoutImageTree["children"][1]["h"] * scaleFixer
  # lookoutImageTreeHD["children"][1]["children"][0]["h"] = lookoutImageTree["children"][1]["children"][0]["h"] * scaleFixer
  # panorama = stitchImageTree(lookoutImageTreeHD, meanBlend = True)

  print(f"Done end to end in {(time.time() - startTime) * 1000}ms")

  cv.imwrite("out.jpg", panorama)
  cv.waitKey(0)