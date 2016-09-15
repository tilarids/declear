import cv2
import sys
import os
import numpy as np

def _line_intersection(l1, l2):
    line1 = ((l1[1], l1[2]), (l1[3],l1[4]))
    line2 = ((l2[1], l2[2]), (l2[3],l2[4]))
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def extract_borders(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines is None or len(lines) == 0:
        return None, None, None, None
    out_lines = []
    height, width = img.shape[0], img.shape[1]
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        out_lines.append((theta,x1,y1,x2,y2))

    left_border = None
    top_border = None
    right_border = None
    bottom_border = None

    for (theta,x1,y1,x2,y2) in out_lines:
        if abs(theta - np.pi / 2) < np.pi / 10: # horizontal line.
            lint = _line_intersection((theta,x1,y1,x2,y2), (0, 0, 0, 0, height))
            rint = _line_intersection((theta,x1,y1,x2,y2), (0, width, 0, width, height))
            if (not top_border or top_border[2] > lint[1]) and lint[1] > 0.02 * height and rint[1] > 0.01 * height:
                top_border = (theta,lint[0],lint[1],rint[0],rint[1])
            if (not bottom_border or bottom_border[2] < lint[1]) and lint[1] < 0.98 * height and rint[1] < 0.98 * height:
                bottom_border = (theta,lint[0],lint[1],rint[0],rint[1])
        elif abs(theta) < np.pi / 10: # vertical line
            tint = _line_intersection((theta,x1,y1,x2,y2), (0, 0, 0, width, 0))
            bint = _line_intersection((theta,x1,y1,x2,y2), (0, 0, height, width, height))
            if (not left_border or left_border[1] > tint[0]) and tint[0] > 0.02 * width and bint[0] > 0.02 * width:
                left_border = (theta, tint[0], tint[1], bint[0], bint[1])
            if (not right_border or right_border[1] < tint[0]) and tint[0] < 0.98 * width and bint[0] < 0.98 * width:
                right_border = (theta,tint[0], tint[1], bint[0], bint[1])

    return left_border, top_border, right_border, bottom_border

def _order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = _order_points(pts)
    (tl, tr, br, bl) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def get_detected_width_and_height(left_border, top_border, right_border, bottom_border):
    detected_width = right_border[1] - left_border[1]
    detected_height = bottom_border[2] - top_border[2]
    return detected_width, detected_height

def extract_fields_given_borders(img, left_border, top_border, right_border, bottom_border):
    if left_border is None or right_border is None or top_border is None or bottom_border is None:
        return None
    height, width = img.shape[0], img.shape[1]
    detected_width, detected_height = get_detected_width_and_height(left_border, top_border, right_border, bottom_border)

    left_border = map(int, list(left_border))
    right_border = map(int, list(right_border))
    top_border = map(int, list(top_border))
    bottom_border = map(int, list(bottom_border))

    WIDTH_THRESHOLD_MIN, WIDTH_THRESHOLD_MAX = (0.75, 0.86)
    HEIGHT_THRESHOLD_MIN, HEIGHT_THRESHOLD_MAX = (0.8, 0.97)
    record = {}

    record['threshold_met'] = (HEIGHT_THRESHOLD_MIN < float(detected_height) / height < HEIGHT_THRESHOLD_MAX and
                               WIDTH_THRESHOLD_MIN < float(detected_width) / width < WIDTH_THRESHOLD_MAX)

    tl = _line_intersection(left_border, top_border)
    tr = _line_intersection(right_border, top_border)
    bl = _line_intersection(left_border, bottom_border)
    br = _line_intersection(right_border, bottom_border)
    warp = four_point_transform(img, np.array([tl, tr, br, bl]))
    wh, ww = warp.shape[0], warp.shape[1]

    TL1 = (0.615866388308977 * ww, 0.13972602739726028 * wh)
    TR1 = (0.7891440501043842 * ww, 0.13972602739726028 * wh)
    BL1 = (0.615866388308977 * ww, 0.16712328767123288 * wh)
    BR1 = (0.7891440501043842 * ww, 0.16712328767123288 * wh)

    TL2 = (0.7933194154488518 * ww, 0.13972602739726028 * wh)
    TR2 = (0.9665970772442589 * ww, 0.13972602739726028 * wh)
    BL2 = (0.7933194154488518 * ww, 0.16712328767123288 * wh)
    BR2 = (0.9665970772442589 * ww, 0.16712328767123288 * wh)

    for k,v in {'first': np.array([TL1, TR1, BR1, BL1]),
                'second': np.array([TL2, TR2, BR2, BL2])
               }.iteritems():
        im = four_point_transform(warp, v)
        imh,imw = im.shape[0], im.shape[1]
        record[k] = im
        record[k+'_width'] = imw
        record[k+'_height'] = imh

        imbin = cv2.adaptiveThreshold(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (95, 22)) ,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        record[k+'_bin'] = imbin
    return record

def extract_fields(img_path):
  img = cv2.imread(img_path)
  left_border, top_border, right_border, bottom_border = extract_borders(img)
  return extract_fields_given_borders(img, left_border, top_border, right_border, bottom_border)

