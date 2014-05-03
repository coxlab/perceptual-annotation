#!/usr/bin/python
# modified from perceptual_detection_training and perceptual_detection_prediction by sam anthony 3/31/13
# example syntax (some system specific stuff in here):
# python perceptual_detection_distance.py --img_list FDDB-fold-01.txt --output_dir output_chips --output_file file.csv --gt FDDB-fold-01-ellipseList.txt \
#  --model_one ./hinge_loss/fold01.model --model_two ./human_weighted_loss/fold01.model --histeq --sift --size 40x40

import cv2
import sys
import argparse
import re

#sys.path.append('libsvm-3.14/python')
sys.path.append('libsvm-3.16/python')
from svmutil import *

sys.path.append('features/')
from dense_sift import *

# Compare two SVM models against one fold of the fddb set (http://vis-www.cs.umass.edu/fddb/)
# For each positive face noted in the ground truth, report the distance of that face 
# and a corresponding negative chip from the two hyperplanes
def compare_models(img_list, gt, model1, model2, feature_x, feature_y, norm, thresh, output_dir, outfile, compute_sift):

   f = open(img_list)
   filenames = f.read().splitlines()
   f.close()

   f = open(gt)
   raw_gt = f.read().splitlines()
   f.close()

   f = open(outfile,'w+')

   extra_negative = 0

   m1 = svm_load_model(model1)
   m2 = svm_load_model(model2)

   # handle the positives first
   gt_pos = 1
   image_ctr = 1
   image_ctr_neg = 1

   f.write('src,cat,ctr,label_hinge,score_hinge,label_human,score_human,diff\n')

   for entry in filenames:
      # load the image
      im = cv2.imread(entry)
      gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      # try:
      #    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      # except cv2.error as e:
      #    print e
      #    continue


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

         outstr = output_dir + "/positive-chip"

         image_ctr = write_image(sub_face_resized, outstr, image_ctr)
         image_ctr_orig = write_image(sub_face, outstr + '_orig', image_ctr - 1)

         outstr = outstr + str(image_ctr - 1)

         process_image_dsift(outstr + '.jpg', outstr + '.sift', 10, feature_x / 10)
         os.remove('tmp.pgm')
         os.remove('tmp.frame')

         label_hinge, score_hinge = filter(sub_face_resized, m1, thresh, feature_y, feature_x, compute_sift)
         label_human, score_human = filter(sub_face_resized, m2, thresh, feature_y, feature_x, compute_sift)

         diff = score_human - score_hinge

         f.write(entry + ',pos,' + str(image_ctr) + ',' + str(label_hinge) + ',' + str(score_hinge) + ',' + \
            str(label_human) + ',' + str(score_human) + ',' + str(diff) + '\n')


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

            outstr = output_dir + "/negative-chip"

            image_ctr_neg = write_image(neg_chip_resized, outstr, image_ctr_neg)
            image_ctr_neg_orig = write_image(neg_chip, outstr + '_orig', image_ctr - 1)

            outstr = outstr + str(image_ctr_neg - 1)

            process_image_dsift(outstr + '.jpg', outstr + '.sift', 10, feature_x / 10)
            os.remove('tmp.pgm')
            os.remove('tmp.frame')

            label_hinge_neg, score_hinge_neg = filter(neg_chip_resized, m1, thresh, feature_y, feature_x, compute_sift)
            label_human_neg, score_human_neg = filter(neg_chip_resized, m2, thresh, feature_y, feature_x, compute_sift)

            diff_neg = score_human_neg - score_hinge_neg

            f.write(entry + ',neg,' + str(image_ctr_neg) + ',' + str(label_hinge_neg) + ',' + str(score_hinge_neg) + ',' + \
               str(label_human_neg) + ',' + str(score_human_neg) + ',' + str(diff_neg) + '\n')

            # if we need an extra negative, do another
            if extra_negative > 0:
               extra_negative -= 1
               j -= 1 
         else:
            extra_negative += 1

         j += 1

   f.close()

def compare_models_tmb(img_list, mask_list, model1, model2, feature_x, feature_y, norm, thresh, output_dir, outfile, compute_sift):
# no gt as an argument because this function assumes that you're passing a list of tmb images that have top and left in the filename
# and equivalent masks in the mask_list arg

   f = open(img_list)
   filenames = f.read().splitlines()
   f.close()

   f = open(mask_list)
   raw_mask = f.read().splitlines()
   f.close()

   f = open(outfile,'w+')

   extra_negative = 0

   m1 = svm_load_model(model1)
   m2 = svm_load_model(model2)

   # handle the positives first
   gt_pos = 1
   image_ctr = 1
   image_ctr_neg = 1

   fn_re = re.compile(r'[0-9\.]+')

   f.write('src,cat,ctr,label_hinge,score_hinge,label_human,score_human,diff\n')

   for index, entry in enumerate(filenames):
      # load the image
      im = cv2.imread(entry)
      mask = cv2.imread(raw_mask[index])

      fn_fields = entry.split('/')
      fn_nums = fn_re.findall(fn_fields[-1])

      gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      # try:
      #    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      # except cv2.error as e:
      #    print e
      #    continue


      # optional normalization
      if norm:
         gray_im = cv2.equalizeHist(gray_im)
     
      full_h, full_w = gray_im.shape

 
      # find the ground truth for the face(s) in this image,
      # then extract its pixels
      x = round(float(fn_nums[-2]))
      y = round(float(fn_nums[-1]))
      h, w, d = mask.shape

      if x < 0:
         x = 0

      if y < 0:
         y = 1
   
      if w > full_w:
         w = full_w

      if h > full_h:
         h = full_h

      # isolate chip and resize to target specification
      sub_face = gray_im[y:y+h, x:x+w]
      sub_face_resized = cv2.resize(sub_face, (feature_x, feature_y))

      outstr = output_dir + "/positive-chip"

      image_ctr = write_image(sub_face_resized, outstr, image_ctr)
      image_ctr_orig = write_image(sub_face, outstr + '_orig', image_ctr - 1)

      outstr = outstr + str(image_ctr - 1)

      process_image_dsift(outstr + '.jpg', outstr + '.sift', 10, feature_x / 10)
      os.remove('tmp.pgm')
      os.remove('tmp.frame')

      label_hinge, score_hinge = filter(sub_face_resized, m1, thresh, feature_y, feature_x, compute_sift)
      label_human, score_human = filter(sub_face_resized, m2, thresh, feature_y, feature_x, compute_sift)

      diff = score_human - score_hinge

      f.write(entry + ',pos,' + str(image_ctr) + ',' + str(label_hinge) + ',' + str(score_hinge) + ',' + \
         str(label_human) + ',' + str(score_human) + ',' + str(diff) + '\n')

   f.close()

def write_image(image, output_dir_header, pos):
   filename = output_dir_header + str(pos) + ".jpg"
   pos += 1
   cv2.imwrite(filename, image)

   return pos

def filter(sub_face_resized, m, thresh, feature_y, feature_x, compute_sift):

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
   else:
      # flatten chip one vector
      for j in range(feature_y):
         for k in range(feature_x):
            image_vector.append(sub_face_resized[j, k])

   p_label, p_acc, p_val = svm_predict([1], [image_vector], m)
   pos_detection = 0

   if p_val[0][0] >= float(thresh):
      pos_detection = 1

   return (pos_detection, p_val[0][0]) 

def main():
   parser = argparse.ArgumentParser(description='Deep Annotation Training Program')
 
   parser.add_argument('--img_list', action="store", dest="img_list", help="list of image filenames")
   parser.add_argument('--output_dir', action="store", dest="output_dir", help ="output directory")
   parser.add_argument('--output_file', action="store", dest="outfile", help ="output CSV file")
   parser.add_argument('--gt', action="store", dest="gt", help="ground truth file")
   parser.add_argument('--model_one', action="store", dest="model1", help ="svm model file one")
   parser.add_argument('--model_two', action="store", dest="model2", help ="svm model file two")
   parser.add_argument('--histeq', action="store_const", dest="norm", const=1, help ="histogram equalization")
   parser.add_argument('--size', action="store", dest="size", help ="face size (WxH)")
   parser.add_argument('--threshold', action="store", dest="thresh", type=float, default=0, help ="svm threshold")
   parser.add_argument('--tmb', action="store_const", dest="tmb", const=1, help ="use testmybrain faces")
   parser.add_argument('--sift', action="store_const", dest="sift", const=1, help ="compute dense sift features")

   args = parser.parse_args()

   fields = str(args.size).split('x')
   feature_x = int(fields[0])
   feature_y = int(fields[1])
   
   if args.tmb:
      #def compare_models_tmb(img_list, mask_list, model1, model2, feature_x, feature_y, norm, thresh, output_dir, outfile, compute_sift)
      compare_models_tmb(args.img_list, args.gt, args.model1, args.model2, feature_x, feature_y, args.norm, args.thresh, args.output_dir,\
         args.outfile, args.sift)
   else:
      #def compare_models(img_list, gt, m1, m2, feature_x, feature_y, norm, thresh, output_dir, compute_sift)
      compare_models(args.img_list, args.gt, args.model1, args.model2, feature_x, feature_y, args.norm, args.thresh, args.output_dir,\
         args.outfile, args.sift)


 #  face_detect(args.img_list, args.output_dir, args.scale, args.neighbors, args.model, args.norm, \
  #             feature_x, feature_y, args.thresh, args.sift)
 
if __name__ == "__main__":
   main()
