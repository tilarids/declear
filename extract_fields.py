import cv2
import sys
import os
import numpy as np

def extract_borders(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,50,150,apertureSize = 3)

  lines = cv2.HoughLines(edges,1,np.pi/180,200)
  out_lines = []
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
      # if abs(theta - np.pi / 2) < np.pi / 10: # horizontal line.
      #   cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
      # elif abs(theta) < np.pi / 10: # vertical line
      #   cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

  left_border = None
  top_border = None
  right_border = None
  bottom_border = None

  for (theta,x1,y1,x2,y2) in out_lines:
    if abs(theta - np.pi / 2) < np.pi / 10: # horizontal line.
      if not top_border or top_border[2] > y1:
        top_border = (theta,x1,y1,x2,y2)
      if not bottom_border or bottom_border[2] < y1:
        bottom_border = (theta,x1,y1,x2,y2)
    elif abs(theta) < np.pi / 10: # vertical line
      if not left_border or left_border[1] > x1:
        left_border = (theta,x1,y1,x2,y2)
      if not right_border or right_border[1] < x1:
        right_border = (theta,x1,y1,x2,y2)

  cv2.line(img,(left_border[1],left_border[2]),(left_border[3],left_border[4]),(0,0,255),2)
  cv2.line(img,(top_border[1],top_border[2]),(top_border[3],top_border[4]),(0,255,0),2)
  cv2.line(img,(right_border[1],right_border[2]),(right_border[3],right_border[4]),(255,0,0),2)
  cv2.line(img,(bottom_border[1],bottom_border[2]),(bottom_border[3],bottom_border[4]),(255,0,255),2)
  return left_border, top_border, right_border, bottom_border


img = cv2.imread(sys.argv[1])
height, width = img.shape[0], img.shape[1]
left_border, top_border, right_border, bottom_border = extract_borders(img)

detected_width = right_border[1] - left_border[1]
detected_height = bottom_border[2] - top_border[2]

WIDTH_THRESHOLD_MIN, WIDTH_THRESHOLD_MAX = (0.7, 0.86)
HEIGHT_THRESHOLD_MIN, HEIGHT_THRESHOLD_MAX = (0.62, 0.77)
# import pdb; pdb.set_trace()
# front page, 2015:
if (HEIGHT_THRESHOLD_MIN < float(detected_height) / height < HEIGHT_THRESHOLD_MAX and
    WIDTH_THRESHOLD_MIN < float(detected_width) / width < WIDTH_THRESHOLD_MAX):
  print "Good!"
else:
  print "Bad!"
# import pdb; pdb.set_trace()
# minLineLength = 200
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,1,np.pi/180, 80, minLineLength, maxLineGap)

# def cmp_line_length(ln1, ln2):
#   return (ln1[0]-ln1[2])**2 + (ln1[1]-ln1[3])**2 - (ln2[0]-ln2[2])**2 - (ln2[1]-ln2[3])**2
# for x1,y1,x2,y2 in sorted(lines[0], cmp=cmp_line_length, reverse=True)[:10]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('/tmp/lines.png', img)
