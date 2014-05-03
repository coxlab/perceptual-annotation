#!/usr/bin/python

import cv2
import sys
import argparse
from multiprocessing.pool import ThreadPool

sys.path.append('libsvm-3.14/python')
from svmutil import *

sys.path.append('features/')
from dense_sift_demo import *

import random

# system specific
cascade_fn = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"
#cascade_fn = "/usr/share/opencv/haarcascades/haarcascade_profileface.xml"
#cascade_fn = "cascades/lbpcascade_profileface.xml"

def face_detect(scale, neighbors, model, norm, feature_x, \
                feature_y, thresh, compute_sift):

   cascade = cv2.CascadeClassifier(cascade_fn)

   cv2.namedWindow("preview")
   vc = cv2.VideoCapture(0)

   if vc.isOpened(): # try to get the first frame
      rval, im = vc.read()
   else:
      rval = False

   m = svm_load_model(model)

   while rval:

      # load the image
      im_vj_orig = im.copy()
      im_detections = im.copy()

      gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      gray = cv2.equalizeHist(gray)

      # optional normalization
      if norm:
         gray = cv2.equalizeHist(gray)

      # step 1: viola-jones at a handful of scales to find candidate faces
      rects = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
      rects_orig = cascade.detectMultiScale(gray)

      for x, y, width, height in rects_orig:
         cv2.rectangle(im_vj_orig, (x, y), (x+width, y+height), (0,0,255), 2)

      positive_detections = []
      rect_index = 0
      print len(rects)

      target_indies = []
#      if len(rects) > 2:
#         target_indices = random.sample(range(len(rects)), 2)
#      elif len(rects) == 1 or len(rects) == 2:
#         target_indices = [0, 1]

      target_indices = [0, 1]

      while rect_index < 2 and rect_index < len(rects):
         # threadpool
         pool = ThreadPool(processes=2)

         x, y, width, height = rects[target_indices[0]]
         
         # isolate chip and resize to target specification
         sub_face = gray[y:y+height, x:x+width]
         sub_face_resized = cv2.resize(sub_face, (feature_x, feature_y))

         # step 2: SVM detector to filter candiates faces
         async_result1 = pool.apply_async(filter, (1, sub_face_resized, m, thresh, feature_y, feature_x, compute_sift))
         if len(rects) > 1 and (rect_index + 1) < len(rects):
            x2, y2, width2, height2 = rects[target_indices[1]]

            # isolate chip and resize to target specification
            sub_face2 = gray[y2:y2+height2, x2:x2+width2]
            sub_face_resized2 = cv2.resize(sub_face2, (feature_x, feature_y))

            # step 2: SVM detector to filter candiates faces
            async_result2 = pool.apply_async(filter, (2, sub_face_resized2, m, thresh, feature_y, feature_x, compute_sift))

         pool.close()
         pool.join()

         label, score = async_result1.get()
         if label == 1:
            positive_detections.append((x,y,width,height,score))

         if len(rects) > 1 and (rect_index + 1) < len(rects):
            label, score2 = async_result2.get()
            if label == 1:
               positive_detections.append((x2,y2,width2,height2,score2))

         rect_index += 2

      positive_detections_sorted = sorted(positive_detections, key=lambda tup: tup[0])
      best_pos = 0
      best_area = -1
      cumulative_score = 0
      best_candidates = []

      neighborhoods = []
      neighborhood_count = 0

      for i in range(len(positive_detections_sorted)):

         curr_x = positive_detections_sorted[i][0]
         curr_y = positive_detections_sorted[i][1]
         curr_width = positive_detections_sorted[i][2]
         curr_height = positive_detections_sorted[i][3]
         curr_score = positive_detections_sorted[i][4]

         neighborhood = -1
         # check if this neighborhood exists, if so, add this entry to it
         for j in range(len(neighborhoods)): 
         
            if neighborhood == -1:
               for k in range(len(neighborhoods[j])):
                  if (neighborhoods[j][k][0] - 30) <= curr_x and\
                     curr_x <= (neighborhoods[j][k][0] + 30) and\
                     (neighborhoods[j][k][1] - 30) <= curr_y and\
                     curr_y <= (neighborhoods[j][k][1] + 30) and\
                     (neighborhoods[j][k][0] + neighborhoods[j][k][2] + 30) >= (curr_x + curr_width) and\
                     (curr_x + curr_width) >= (neighborhoods[j][k][0] + neighborhoods[j][k][2] - 30) and\
                     (neighborhoods[j][k][1] + neighborhoods[j][k][3] + 30) >= (curr_y + curr_height) and\
                     (curr_y + curr_height) >= (neighborhoods[j][k][1] + neighborhoods[j][k][3] - 30):
                     neighborhoods[j].append((curr_x,curr_y,curr_width,curr_height,curr_score))
                     neighborhood = j
                     break
            else:
               break

         # if we haven't seen this neighborhood before, add it
         if neighborhood == -1:

            neighborhoods.append([])
            neighborhoods[neighborhood_count].append((curr_x,curr_y,curr_width,curr_height,curr_score))
            neighborhood_count += 1

      best_candidates = []

      for i in range(len(neighborhoods)):

         neighborhood_score = 0
         best_index = -1
         best_score = -1

         for j in range(len(neighborhoods[i])):
            neighborhood_score += neighborhoods[i][j][4]
            if neighborhoods[i][j][4] > best_score:
               best_score = neighborhoods[i][j][4]
               best_index = j

         best_candidates.append((neighborhoods[i][best_index][0],neighborhoods[i][best_index][1],neighborhoods[i][best_index][2],neighborhoods[i][best_index][3],neighborhood_score))

      results_for_scoring = []
      for i in range(len(best_candidates)):

         curr_x = best_candidates[i][0]
         curr_y = best_candidates[i][1] - int(best_candidates[i][3] / 10)
         if curr_y <= 0:
            curr_y = 1

         curr_x_lower_right = best_candidates[i][0] + best_candidates[i][2]
         curr_y_lower_right = best_candidates[i][1] + best_candidates[i][3] + int(best_candidates[i][3] / 10)
         im_height, im_width = gray.shape
         if curr_y_lower_right >= im_height:
            curr_y_lower_right = im_height - 1

         curr_y_score = best_candidates[i][4]

         cv2.rectangle(im_detections, (curr_x, curr_y), (curr_x_lower_right,\
                       curr_y_lower_right), (0,255,0), 2)

      cv2.imshow("perceptual annotation", im_detections)
      cv2.imshow("viola-jones", im_vj_orig)
      rval, im = vc.read()
      key = cv2.waitKey(1)
      if key == 27: # exit on ESC
         break

def filter(tag, sub_face_resized, m, thresh, feature_y, feature_x, compute_sift):

   image_vector = []
   if compute_sift:
      cv2.imwrite('/media/demodisk/temp-image0' + str(tag) + '.jpg', sub_face_resized)
      process_image_dsift(tag, '/media/demodisk/temp-image0' + str(tag) + '.jpg', '/media/demodisk/temp-image0' + str(tag) + '.sift', 10, feature_x / 10)
      l,d = read_features_from_file('/media/demodisk/temp-image0' + str(tag) + '.sift')
      image_vector = d.flatten().tolist()
      os.remove('/media/demodisk/temp-image0' + str(tag) + '.jpg')
      os.remove('/media/demodisk/temp-image0' + str(tag) + '.sift')
      os.remove('/media/demodisk/tmp' + str(tag) + '.pgm')
      os.remove('/media/demodisk/tmp' + str(tag) + '.frame')
   else:
      # flatten chip one vector
      for j in range(feature_y):
         for k in range(feature_x):
            image_vector.append(sub_face_resized[j, k])

   p_label, p_acc, p_val = svm_predict([1], [image_vector], m)
   pos_detection = 0
   if p_val[0][0] >= float(thresh):
      pos_detection = 1

   return pos_detection, p_val[0][0]

def main():
   parser = argparse.ArgumentParser(description='Deep Annotation Training Program')
 
   parser.add_argument('--model', action="store", dest="model", help ="svm model file")
   parser.add_argument('--scale', action="store", dest="scale", type=float, default=1.1, help ="scale factor")
   parser.add_argument('--neighbors', action="store", dest="neighbors", type=int, default=0, help ="min neighbors for detection")
   parser.add_argument('--histeq', action="store_const", dest="norm", const=1, help ="histogram equalization")
   parser.add_argument('--size', action="store", dest="size", help ="face size (WxH)")
   parser.add_argument('--threshold', action="store", dest="thresh", type=float, default=0, help ="svm threshold")
   parser.add_argument('--sift', action="store_const", dest="sift", const=1, help ="compute dense sift features")

   args = parser.parse_args()

   fields = str(args.size).split('x')
   feature_x = int(fields[0])
   feature_y = int(fields[1])
   
   face_detect(args.scale, args.neighbors, args.model, args.norm, feature_x, feature_y, args.thresh, args.sift)
 
if __name__ == "__main__":
   main()
