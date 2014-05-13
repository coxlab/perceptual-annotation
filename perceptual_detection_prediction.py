#!/usr/bin/python
#
# Face Detector with Perceptual Annotation 
#
# - Process a data set
# - Select different features and normalizations
# - Produce output that can be scored by FDDB scoring routines 

import cv2
import sys
import glob
import argparse

import numpy as np
import sthor
from sthor.model.slm import SequentialLayeredModel

sys.path.append('libsvm-3.14/python')
from svmutil import *

sys.path.append('features/')
from dense_sift import *

# system specific
cascade_fn = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"

def face_detect(img_list, output_dir, scale, neighbors, model, norm, feature_x, \
                feature_y, thresh, compute_sift, compute_coxlab, results_file):

   f = open(img_list)
   filenames = f.read().splitlines()
   f.close()

   cascade = cv2.CascadeClassifier(cascade_fn)

   f = open(results_file, "w")

   counter = 0
   for entry in filenames:
      fields = entry.split('/')
      image_name = fields[len(fields) - 5] + '/' + fields[len(fields) - 4] +\
                   '/' + fields[len(fields) - 3] + '/' + fields[len(fields) - 2] + '/'+\
                   fields[len(fields) - 1].split('.')[0]

      # load the image
      im = cv2.imread(entry)
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

      m = svm_load_model(model)

      for x, y, width, height in rects_orig:
         cv2.rectangle(im_vj_orig, (x, y), (x+width, y+height), (255,0,0), 2)

      positive_detections = []
      for x, y, width, height in rects:
         cv2.rectangle(im, (x, y), (x+width, y+height), (255,0,0), 2)
         
         # isolate chip and resize to target specification
         sub_face = gray[y:y+height, x:x+width]
         sub_face_resized = cv2.resize(sub_face, (feature_x, feature_y))

         # step 2: SVM detector to filter candiates faces
         label, score = filter(sub_face_resized, m, thresh, feature_y, feature_x, compute_sift, compute_coxlab)
         if label == 1:
            positive_detections.append((x,y,width,height,score))

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
                       curr_y_lower_right), (255,0,0), 2)
         results_for_scoring.append(str(best_candidates[i][0] + int(best_candidates[i][2] / 10))
                                    + ' ' + str(curr_y)\
                                    + ' ' + str(best_candidates[i][2] - int((best_candidates[i][2] / 10) * 2))\
                                    + ' ' +  str(best_candidates[i][3] + (int(best_candidates[i][3] / 10) * 2)) +\
                                    ' ' + str(best_candidates[i][4]))

      f.write(image_name + '\n')
      f.write(str(len(results_for_scoring)) + '\n')
      for i in range(len(results_for_scoring)):
         f.write(results_for_scoring[i] + '\n')

      if output_dir is not None:
         out_str_vj_default = output_dir + "/test-viola-jones-default" + str(counter) + ".jpg"
         out_str_vj = output_dir + "/test-viola-jones" + str(counter) + ".jpg"
         out_str_detect = output_dir + "/test-detect" + str(counter) + ".jpg"
         cv2.imwrite(out_str_vj_default, im_vj_orig)
         cv2.imwrite(out_str_vj, im)
         cv2.imwrite(out_str_detect, im_detections)

      counter += 1
   f.close()

def filter(sub_face_resized, m, thresh, feature_y, feature_x, compute_sift, compute_coxlab):

   image_vector = []
   if compute_sift:
      cv2.imwrite('./temp-image0.jpg', sub_face_resized)
      process_image_dsift('./temp-image0.jpg', './temp-image0.sift', 10, feature_x / 10)
      l,d = read_features_from_file('./temp-image0.sift')
      image_vector = d.flatten().tolist()
      os.remove('temp-image0.jpg')
      os.remove('temp-image0.sift')
      os.remove('tmp.pgm')
      os.remove('tmp.frame')
   elif compute_coxlab:
      image_vector = gen_coxlab_features(sub_face_resized, feature_x, feature_y)
   else:
      # flatten chip one vector
      for j in range(feature_y):
         for k in range(feature_x):
            image_vector.append(sub_face_resized[j, k])

   p_label, p_acc, p_val = svm_predict([1], [image_vector], m)
   pos_detection = 0

   p_val[0][0] += abs(float(thresh))

   if p_val[0][0] >= 0:
      pos_detection = 1

   return (pos_detection, p_val[0][0]) 

# generate coxlab features
def gen_coxlab_features(chip_resized, feature_x, feature_y):

   image_vector = []

   im_array = np.asarray(chip_resized).astype('f')

   # -- get L3 prime SLM description (from sthor)
   desc = sthor.model.parameters.fg11.fg11_ht_l3_1_description

   # -- generate random PLOS09 3-layer model
   #desc = sthor.model.parameters.plos09.get_random_plos09_l3_description()

   # -- instantiate the SLM model
   model = SequentialLayeredModel((feature_x, feature_y), desc)

   # -- compute feature map, shape [height, width, depth]
   f_map = model.transform(im_array, pad_apron=True, interleave_stride=False)
   f_map_dims = f_map.shape
   print "shape", f_map.shape

   for j in range(f_map_dims[0]):
      for k in range(f_map_dims[1]):
         for l in range(f_map_dims[2]):
            image_vector.append(f_map[j][k][l])

   return image_vector

def main():
   parser = argparse.ArgumentParser(description='Perceptual Annotation Training Program')
 
   parser.add_argument('--img_list', action="store", dest="img_list", help="list of image filenames")
   parser.add_argument('--output_dir', action="store", dest="output_dir", help ="output directory")
   parser.add_argument('--model', action="store", dest="model", help ="svm model file")
   parser.add_argument('--scale', action="store", dest="scale", type=float, default=1.1, help ="scale factor")
   parser.add_argument('--neighbors', action="store", dest="neighbors", type=int, default=0, help ="min neighbors for detection")
   parser.add_argument('--histeq', action="store_const", dest="norm", const=1, help ="histogram equalization")
   parser.add_argument('--size', action="store", dest="size", help ="face size (WxH)")
   parser.add_argument('--threshold', action="store", dest="thresh", type=float, default=0, help ="svm threshold")
   parser.add_argument('--sift', action="store_const", dest="sift", const=1, default=0, help ="compute dense sift features")
   parser.add_argument('--coxlab', action="store_const", dest="coxlab", const=1, default=1, help ="compute coxlab features")
   parser.add_argument('--results', action="store", dest="results_file", help ="results file")

   args = parser.parse_args()

   fields = str(args.size).split('x')
   feature_x = int(fields[0])
   feature_y = int(fields[1])

   face_detect(args.img_list, args.output_dir, args.scale, args.neighbors, args.model, args.norm, \
               feature_x, feature_y, args.thresh, args.sift, args.coxlab, args.results_file)
 
if __name__ == "__main__":
   main()
