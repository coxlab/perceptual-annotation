Face Detection Demo
===========================

A work in progress...
---------------------

Runs in *almost* real-time on a sufficiently fast computer.

    Use a directory mapped to memory for temporary files:
    mkdir /media/demodisk
    mount -t tmpfs none /media/demodisk

    Add dense_sift_demo.py to features/ directory

Sample command:

    Face Detection with Dense SIFT Features:
    python webcam_demo.py --model fddb.model --scale 1.05 --sift --histeq --size 40x40 --threshold 0
