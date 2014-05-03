Perceptual Annotation Source Code
===========================

Dependencies
------------

You will need to have working versions of:

    numpy
    matplotlib
    opencv 2.3 (with python bindings)
    libsvm-3.14
    vlfeat-0.9.16
    sthor

Installation Steps
------------------

Download and build libsvm. It's kept locally for now, as we will make a few modifications to it in the future:

    $ cd perceptual-annotation
    $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/libsvm-3.14.tar.gz
    $ gzip -cd libsvm-3.14.tar.gz | tar xvf -
    $ cd libsvm-3.14
    $ make ; cd python; make

Download vlfeat. It will be necessary to generate SIFT features:

    $ cd perceptual-annotation/features
    $ wget http://www.vlfeat.org/download/vlfeat-0.9.16-bin.tar.gz
    $ gzip -cd libsvm-3.14.tar.gz | tar xvf -

    Create a symlink to the sift binary for your platform:
    Example: $ ln -s ./vlfeat-0.9.16/bin/glnx86/sift sift

Install sthor. Instructions avaiable at https://github.com/nsf-ri-ubicv/sthor

Example Usage
-------------

You will need data from TMB to build models. The sets can be retrieved from:
www.testmybrain.org (for credentials, ask santhony)

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
    python perceptual_detection_training_git_validate.py --sift --output validation. --size 30x30 --num_face 100 --histeq --tmb_images tmb/simo-files --tmb_chips perceptual-annotation/experiments/tmb-masks/tmb_simoncelli_chip_sizes --simo

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
    python perceptual_detection_prediction.py --img_list FDDB/FDDB-folds/FDDB-fold-01-full-path.txt --model fold_01.training.model --histeq --coxlab --size 30x30 --scale 1.05 --threhold -1.5 --results validate.out > /dev/null
