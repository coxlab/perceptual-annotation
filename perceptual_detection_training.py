#!/usr/bin/python
#
# Deep Annotation Feature Generation Program
#
# - Process specific data sets and tmb data
# - Select different features and normalizations
# - All routines needed to generate data from T.PAMI submission
# - ASCII output is suitable for LIBSVM (patched for human weighted loss
#   for deep annotations)

import cv2
import sys
import glob
import argparse
from random import randint

import numpy as np
import sthor
from sthor.model.slm import SequentialLayeredModel

sys.path.append('libsvm-3.14/python')
from svmutil import *

sys.path.append('features/')
from dense_sift import *

# Process data from the UMass FDDB set (http://vis-www.cs.umass.edu/fddb/)
# For each positive face noted in the ground truth, attempt to find a corresponding sample.
def process_fddb(img_list, gt, feature_x, feature_y, output, norm, num_faces, debug,\
                 compute_sift, compute_coxlab):

   f = open(img_list)
   filenames = f.read().splitlines()
   f.close()

   f = open(gt)
   raw_gt = f.read().splitlines()
   f.close()

   flattened_vectors = []
   flattened_vectors_positive = []
   flattened_vectors_negative = []
   extra_negative = 0

   # handle the positives first
   gt_pos = 1
   image_ctr = 1
   image_ctr_neg = 1
   for entry in filenames:
      # load the image
      im = cv2.imread(entry)
      gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

      # optional normalization
      if norm:
         gray_im = cv2.equalizeHist(gray_im)
     
      full_h, full_w = gray_im.shape
 
      # find the ground truth for the face(s) in this image,
      # then extract its pixels
      num_gt_faces = int(raw_gt[gt_pos])
      gt_pos += 1
      coord_table = []
      for i in range(num_gt_faces):
         fields = raw_gt[gt_pos].split()
         x = round(float(fields[3])) - round(float(fields[1]))
         y = round(float(fields[4])) - round(float(fields[0]))
         w = 2 * round(float(fields[1]))
         h = 2 * round(float(fields[0]))

         if x < 0:
            x = 0

         if y < 0:
            y = 1
      
         if w > full_w:
            w = full_w

         if h > full_h:
            h = full_h

         coord_table.append((x,y,w,h))

         # isolate chip and resize to target specification
         sub_face = gray_im[y:y+h, x:x+w]
         sub_face_resized = cv2.resize(sub_face, (feature_x, feature_y))

         if debug:
            image_ctr = write_image(sub_face_resized, debug + "/positive-chip", image_ctr)

         if compute_sift:
            flattened_vectors_positive.append(process_sift(sub_face_resized, feature_x))
         elif compute_coxlab:
            flattened_vectors_positive.append(gen_coxlab_features(sub_face_resized, feature_x, feature_y))
         else:
            # flatten into one vector
            image_vector = []
            for j in range(feature_y):
               for k in range(feature_x):
                  image_vector.append(sub_face_resized[j, k])

            flattened_vectors_positive.append(image_vector)

         gt_pos += 1
      gt_pos += 1

      # now find suitable negatives
      j = 0
      while j < len(coord_table):
         neg_width = coord_table[j][2]
         neg_height = coord_table[j][3]

         # sample a random point
         if full_w - neg_width == 0:
            neg_width = neg_width - 10

         if full_h - neg_height == 0:
            neg_height = neg_height - 10

         neg_x = 0
         neg_y = 0
         if (full_w - neg_width - 1) != 0:
            neg_x = randint(0, full_w - neg_width - 1)
         if (full_h - neg_height - 1) != 0:
            neg_y = randint(0, full_h - neg_height - 1)

         # check the point against the positives;
         # if somewhere close to a real face, try again
         count_tries = 0
         k = 0
         while k < len(coord_table) and count_tries < 10:
            if neg_x >= (coord_table[k][0] - (neg_width / 2)) and neg_x <= (coord_table[k][0] + (neg_width / 2)) and \
               neg_y >= (coord_table[k][1] - (neg_height / 2))  and neg_y <= (coord_table[k][1] + (neg_height / 2)):

               if (full_w - neg_width - 1) != 0:
                  neg_x = randint(0, full_w - neg_width - 1)
               if (full_h - neg_height - 1) != 0:
                  neg_y = randint(0, full_h - neg_height - 1)
               count_tries += 1
               k = 0
            else:
               k += 1
          
         if count_tries < 10:

            neg_chip = gray_im[neg_y:neg_y+neg_height, neg_x:neg_x+neg_width]
            neg_chip_resized = cv2.resize(neg_chip, (feature_x, feature_y))

            if debug:
               image_ctr_neg = write_image(neg_chip_resized, debug + "/negative-chip", image_ctr_neg)

            if compute_sift:
               flattened_vectors_negative.append(process_sift(neg_chip_resized, feature_x))
            elif compute_coxlab:
               flattened_vectors_negative.append(gen_coxlab_features(neg_chip_resized, feature_x, feature_y))
            else:
               # flatten into one vector
               image_vector = []
               for k in range(feature_y):
                  for l in range(feature_x):
                     image_vector.append(neg_chip_resized[k, l])

               flattened_vectors_negative.append(image_vector)

            # if we need an extra negative, do another
            if extra_negative > 0:
               extra_negative -= 1
               j -= 1 
         else:
            extra_negative += 1

         j += 1

      if num_faces:
         if len(flattened_vectors_positive) >= int(num_faces):
            break         

   flattened_vectors = flattened_vectors_positive + flattened_vectors_negative   

   generate_ascii_features(flattened_vectors, output)

# For debugging purposes
def write_image(image, output_file_header, pos):
   filename = output_file_header + str(pos) + ".jpg"
   pos += 1
   cv2.imwrite(filename, image)

   return pos

# Call off to a canonical feature generation program (VLFeat) for SIFT.
def process_sift(chip_resized, feature_x):
   write_image(chip_resized, './temp-image', 0)
   process_image_dsift('./temp-image0.jpg', './temp-image0.sift', 10, feature_x / 10)
   l,d = read_features_from_file('./temp-image0.sift')
   os.remove('temp-image0.jpg')
   os.remove('temp-image0.sift')
   os.remove('tmp.pgm')
   os.remove('tmp.frame')

   return d.flatten()

# Assume balanced features
def generate_ascii_features(raw_vectors, outfile):
   outfile += ".txt"
   f = open(outfile,'w')

   boundary = len(raw_vectors) / 2
   for i in range(len(raw_vectors)):
      if i < boundary:
         vector_str = "+1"
      else:
         vector_str = "-1"
      for j in range(len(raw_vectors[i])):
         vector_index = str(j + 1)
         vector_str += " " + vector_index + ":" + str(raw_vectors[i][j])

      f.write(vector_str)
      f.write("\n")

   f.close()

# TODO
def meanvarpatchnorm(flattened_vectors):
   x = 1

# Use the filename and size table from Sam to locate the face in the TMB image.
   # Use the filename and size table from Sam to locate the face in the TMB image.
def tmb_find_chip(entry, norm, size_table, image_ctr, feature_x, feature_y, debug):
   # load the image
   im = cv2.imread(entry)
   gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

   # optional normalization
   if norm:
      gray_im = cv2.equalizeHist(gray_im)

   # parse the filename
   depth = entry.split("/")
   id = int(depth[len(depth) - 1].split("face")[1].split("no")[0])
   filenum = int(depth[len(depth) - 1].split("no")[1].split("co")[0])
   x = int(depth[len(depth) - 1].split("l")[1].split("t")[0])
   y = int(depth[len(depth) - 1].split("t")[1].split(".png")[0])

   full_h, full_w = gray_im.shape

   # isolate chip and resize to target specification
   if y+size_table[(id, filenum)][1] >= full_h and x+size_table[(id, filenum)][0] >= full_w:
      sub_face = gray_im[y:full_h - 1, x:full_w - 1]
   elif y+size_table[(id, filenum)][1] >= full_h:
      sub_face = gray_im[y:full_h - 1, x:x+size_table[(id,filenum)][0]]
   elif x+size_table[(id, filenum)][0] >= full_w:
      sub_face = gray_im[y:y+size_table[(id, filenum)][1], x:full_w - 1]
   else:
      sub_face = gray_im[y:y+size_table[(id, filenum)][1], x:x+size_table[(id, filenum)][0]]

   sub_face_resized = cv2.resize(sub_face, (feature_x, feature_y))

   if debug:
      image_ctr = write_image(sub_face_resized, debug + "/tmb-chip", image_ctr)

   return sub_face_resized, image_ctr

# Read the face chip sizes from file
def load_tmb_face_chips_sizes(chip_sizes):
   # load the chip sizes
   size_table = {};
   f = open(chip_sizes)
   raw_chip_sizes = f.read().splitlines()
   f.close()

   for entry in raw_chip_sizes:
      fields = entry.split()
      x, y = fields[2].split('x')
      size_table[(int(fields[0]), int(fields[1]))] = (int(x), int(y))
  
   return size_table 

# Generate ASCII features in libsvm format for AFLW TMB images. Assumes half the data is positive and
# half the data is negative.
def gen_tmb_features_aflw(test_files, norm, feature_x, feature_y, output, compute_sift, compute_coxlab, num_faces, debug):
   f = open(test_files)
   filenames = f.read().splitlines()
   f.close()

   image_ctr = 0
   line_ctr = 0
   flattened_vectors_tmb = []

   for entry in filenames:

      # load the image
      im = cv2.imread(entry)
      gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

      # optional normalization
      if norm:
         gray_im = cv2.equalizeHist(gray_im)

      im_resized = cv2.resize(gray_im, (feature_x, feature_y))

      if debug:
         if num_faces:
            if line_ctr < int(num_faces):
               image_ctr = write_image(im_resized, debug + "/tmb-chip", image_ctr)
            else:
               image_ctr = write_image(im_resized, debug + "/tmb-neg-chip", image_ctr)
         else:
            if line_ctr < len(filenames)/2:
               image_ctr = write_image(im_resized, debug + "/tmb-chip", image_ctr)
            else:
               image_ctr = write_image(im_resized, debug + "/tmb-neg-chip", image_ctr)

      if compute_sift:
         flattened_vectors_tmb.append(process_sift(im_resized, feature_x))
      elif compute_coxlab:
         flattened_vectors_tmb.append(gen_coxlab_features(im_resized, feature_x, feature_y))
      else:
         # flatten into one vector
         image_vector = []
         for j in range(feature_y):
            for k in range(feature_x):
               image_vector.append(im_resized[j, k])

         flattened_vectors_tmb.append(image_vector)

      line_ctr += 1

      if num_faces:
         if len(flattened_vectors_tmb) >= int(num_faces) * 2:
            break

   generate_ascii_features(flattened_vectors_tmb, output + 'tmb-features')

# Generate ASCII features in libsvm format for Portilla-Simoncelli TMB images. Assumes half the data
# is positive and half the data is negative.
def gen_tmb_features_simo(test_files, chip_sizes, norm, feature_x, feature_y, output, compute_sift, compute_coxlab,\
                          num_faces, debug):
   f = open(test_files)
   filenames = f.read().splitlines()
   f.close()

   image_ctr = 0
   line_ctr = 0
   flattened_vectors_tmb = []

   size_table = load_tmb_face_chips_sizes(chip_sizes)

   for entry in filenames:

      # special parsing for positive tmb faces
      if num_faces:
         if line_ctr < int(num_faces):
            im_resized, image_ctr = tmb_find_chip(entry, tmb_find_chip, size_table, image_ctr,\
                                                  feature_x, feature_y, debug)
      elif line_ctr < len(filenames)/2:
         im_resized, image_ctr = tmb_find_chip(entry, tmb_find_chip, size_table, image_ctr,\
                                               feature_x, feature_y, debug)
      else:
         # load the image
         im = cv2.imread(entry)
         gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

         # optional normalization
         if norm:
            gray_im = cv2.equalizeHist(gray_im)

         im_resized = cv2.resize(gray_im, (feature_x, feature_y))

         if debug:
            image_ctr = write_image(im_resized, debug + "/tmb-neg-chip", image_ctr)

      if compute_sift:
         flattened_vectors_tmb.append(process_sift(im_resized, feature_x))
      elif compute_coxlab:
         flattened_vectors_tmb.append(gen_coxlab_features(im_resized, feature_x, feature_y))
      else:
         # flatten into one vector
         image_vector = []
         for j in range(feature_y):
            for k in range(feature_x):
               image_vector.append(im_resized[j, k])

         flattened_vectors_tmb.append(image_vector)

      line_ctr += 1

      if num_faces:
         if len(flattened_vectors_tmb) >= int(num_faces) * 2:
            break

   generate_ascii_features(flattened_vectors_tmb, output + 'tmb-features')

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
   parser = argparse.ArgumentParser(description='Deep Annotation Training Program')
 
   parser.add_argument('--img_list', action="store", dest="img_list", help="list of image filenames")
   parser.add_argument('--gt', action="store", dest="gt", help="ground truth file")
   parser.add_argument('--output', action="store", dest="output", help ="model output basename")
   parser.add_argument('--debug', action="store", dest="debug", help ="debug output directory")
   parser.add_argument('--size', action="store", dest="size", help ="face size (WxH)")
   parser.add_argument('--num_face', action="store", dest="num_face", help="number of faces")
   parser.add_argument('--histeq', action="store_const", dest="norm", const=1, help ="histogram equalization")
   parser.add_argument('--sift', action="store_const", dest="sift", const=1, default=0, help ="compute dense sift features")
   parser.add_argument('--aflw', action="store_const", dest="aflw", const=1, help ="consider aflw tmb data")
   parser.add_argument('--simo', action="store_const", dest="simo", const=1, help ="consider portilla-simoncelli tmb data")
   parser.add_argument('--tmb_images', action="store", dest="tmb", help ="list of tmb images")
   parser.add_argument('--tmb_chips', action="store", dest="tmb_chips", help ="chip sizes for tmb images")
   parser.add_argument('--fddb', action="store_const", dest="fddb", const=1, help="process fddb data (default)")
   parser.add_argument('--coxlab', action="store_const", dest="coxlab", const=1, default=1, help="compute coxlab features (default)")

   args = parser.parse_args()

   fields = str(args.size).split('x')
   feature_x = int(fields[0])
   feature_y = int(fields[1])

   if args.fddb:
      process_fddb(args.img_list, args.gt, feature_x, feature_y, args.output, args.norm, args.num_face, args.debug,\
                   args.sift, args.coxlab)
   elif args.tmb and args.aflw:
      gen_tmb_features_aflw(args.tmb, args.norm, feature_x, feature_y, args.output, args.sift, args.coxlab, args.num_face,\
                            args.debug)
   elif args.tmb and args.simo:
      gen_tmb_features_simo(args.tmb, args.tmb_chips, args.norm, feature_x, feature_y, args.output, args.sift, args.coxlab,\
                            args.num_face, args.debug)
 
if __name__ == "__main__":
   main()
