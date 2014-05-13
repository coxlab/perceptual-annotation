Perceptual Annotation Source Code
===========================

For many problems in computer vision, human learners are considerably better than machines. Humans possess
highly accurate internal recognition and learning mechanisms that are not yet understood, and they frequently have access
to more extensive training data through a lifetime of unbiased experience with the visual world. We propose to use visual
psychophysics to directly leverage the abilities of human subjects to build better machine learning systems. First, we use an
advanced online psychometric testing platform to make new kinds of annotation data available for learning. Second, we
develop a technique for harnessing these new kinds of information – "perceptual annotations" – for support vector machines. A key
intuition for this approach is that while it may remain infeasible to dramatically increase the amount of data and high-quality
labels available for the training of a given system, measuring the exemplar-by-exemplar difﬁculty and pattern of errors of human 
annotators can provide important information for regularizing the solution of the system at hand. 

This repository contains reference code for an SVM-variant that makes use of perceptual annotations, and a face detector built
arounnd the core notion.

Dependencies
------------

You will need to have working versions of:

    numpy
    matplotlib
    opencv 2.3 (with python bindings)
    libsvm-3.14
    vlfeat-0.9.16
    sthor

Data
----

Example features and perceptual annotations from the TestMyBrain.org website are available for reference.
The following archives can be downloaded:

Portilla-Simoncelli Images and Perceptual Annotations
http://www.wjscheirer.com/datasets/perceptual_annotation/portilla_simoncelli.tgz

Biologically-inspired Features (FDDB and AFLW) and Perceptual Annotations (AFLW)
http://www.wjscheirer.com/datasets/perceptual_annotation/features_biologically_inspired.tgz

HOG Features (FDDB and Portilla-Simoncelli Set) and Perceptual Annotations (Portilla-Simoncelli Set)
http://www.wjscheirer.com/datasets/perceptual_annotation/features_HOG.tgz

Installation Steps
------------------

Download and build libsvm. It's kept locally for now, as we will make a few modifications to it in the future:

    $ cd perceptual-annotation
    $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/libsvm-3.14.tar.gz
    $ gzip -cd libsvm-3.14.tar.gz | tar xvf -
    $ cd libsvm-3.14
    $ patch < ../libsvm_human_weighted_loss.patch
    $ patch < ../../libsvm_svmutil.py.patch
    $ make ; cd python; make

Download vlfeat. It will be necessary to generate SIFT features:

    $ cd perceptual-annotation/features
    $ wget http://www.vlfeat.org/download/vlfeat-0.9.16-bin.tar.gz
    $ gzip -cd libsvm-3.14.tar.gz | tar xvf -

    Create a symlink to the sift binary for your platform:
    Example: $ ln -s ./vlfeat-0.9.16/bin/glnx86/sift sift

Install sthor. Instructions avaiable at https://github.com/nsf-ri-ubicv/sthor

Example SVM Training Usage 
--------------------------

The only difference between the standard usage of libsvm and this modified version is encountered
at training time. You will need to provide one extra filename argument: the weight file containing
the perceptual annotations for each feature vector. For example, consider the data found in the
features_biologically_inspired.tgz archive. To train a model for the first fold of the FDDB data set 
incorporating some perceptually annotated images from the AFLW data set:

    $ svm-train -c 1 -t 0 fddb-bio-30x30-01.txt aflw_weights.txt fddb-bio-30x30-01.txt.model

The resulting model can be used without any further special considerations.

Example Face Detector Usage
---------------------------

You will need data from TestMyBrain.org to build models. Examples are provided in the data
archives referenced above.

perceptual_detection_training.py generates features to train models from. It has many options:

    -h, --help            show this help message and exit
    --img_list IMG_LIST   list of image filenames
    --gt GT               ground truth file
    --output OUTPUT       model output basename
    --debug DEBUG         debug output directory
    --size SIZE           face size (WxH)
    --num_face NUM_FACE   number of faces
    --histeq              histogram equalization
    --sift                compute dense sift features
    --aflw                consider aflw tmb data
    --simo                consider portilla-simoncelli tmb data
    --tmb_images TMB      list of tmb images
    --tmb_chips TMB_CHIPS chip sizes for tmb images
    --fddb                process fddb data (default)
    --coxlab              compute coxlab features (default)

Sample commands:

    Generate Dense SIFT Features for FDDB Data:
    python perceptual_detection_training_git_validate.py --fddb --img_list FDDB/FDDB-folds/FDDB-fold-01-full-path.txt --gt FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt --output validation.features --size 30x30 --num_face 100 --histeq --sift

    Generate Biologically-Inspired Features for FDDB Data:
    python perceptual_detection_training_git_validate.py --fddb --img_list FDDB/FDDB-folds/FDDB-fold-01-full-path.txt --gt FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt --output validation.features --size 30x30 --num_face 100 --histeq --coxlab

    Generate Dense SIFT Features for Portilla-Simoncelli TMB Data:
    python perceptual_detection_training_git_validate.py --sift --output validation. --size 30x30 --num_face 100 --histeq --tmb_images tmb/simo-files --tmb_chips portilla_simoncelli/tmb_simoncelli_chip_sizes --simo

    Generate Biologically-Inspired Features for Portilla-Simoncelli TMB Data:
    python perceptual_detection_training_git_validate.py --fddb --img_list FDDB/FDDB-folds/FDDB-fold-01-full-path.txt --gt FDDB/FDDB-folds/FDDB-fold-01-ellipseList.txt --output validation. --size 30x30 --num_face 100 --histeq --coxlab

    Generate Dense SIFT Features for AFLW TMB Data:
    python perceptual_detection_training_git_validate.py --sift --output validation. --size 30x30 --num_face 100 --histeq --tmb_images aflw/aflw_files  --aflw

    Generate Biologically-Inspired Features for AFLW TMB Data:
    python perceptual_detection_training_git_validate.py --aflw --output validation. --size 30x30 --num_face 100 --histeq --tmb_images aflw/aflw_files --coxlab

perceptual_detection_prediction.py is the face detection code. It also has many options:

    -h, --help                show this help message and exit
    --img_list IMG_LIST       list of image filenames
    --output_dir OUTPUT_DIR   output directory
    --model MODEL             svm model file
    --scale SCALE             scale factor
    --neighbors NEIGHBORS     min neighbors for detection
    --histeq                  histogram equalization
    --size SIZE               face size (WxH)
    --threshold THRESHOLD     svm threshold
    --sift                    compute dense sift features
    --coxlab                  compute coxlab features
    --results RESULTS         results file

Sample command:

    Face Detection with Biologically-Inspired Features:
    python perceptual_detection_prediction.py --img_list FDDB/FDDB-folds/FDDB-fold-01-full-path.txt --model fddb-bio-30x30-01.txt.model --histeq --coxlab --size 30x30 --scale 1.05 --threhold -1.5 --results validate.out > /dev/null

Citing this Code
----------------

If you use this code in your own work, please cite the following paper:

"Perceptual Annotation: Measuring Human Vision to Improve Computer Vision,"
Walter Scheirer, Samuel E. Anthony, Ken Nakayama, David D. Cox
IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2014.
