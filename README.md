# Human-Motion-Prediction

By Changliu Liu, Yujiao Cheng, Weiye Zhao

### Introduction
**Human-Motion-Prediction** is currently implemented with two different algorithms, RLS-PAA and Identifier-based algorithm.

### Requirements: software

0.	MATLAB 2014a or later.

### Requirements: hardware

CPU, Windows 7 or later, MAC OS.

### Demo
0.	Run `id_demo.m` to apply identifier-based algorithm on human motion data.
0.	Run `rls_demo.m` to apply RLS-PAA algorithm on human motion data.

### Training & Testing
0. Both `id_demo.m` and `rls_demo.m` have training and testing part. Check both files for more details.
    - **Note**: check opts. parameter in demo files for more parameter setting.
0. Check other scripts in `./lib` for auxiliary function.

**Note:** 
- In all the experiments, training is performed on smoothed human motion data`.\data2\trainX&Y`, and testing is performed on smoothed human motion data`.\data2\TestX&Y`. Both training and testin data are stored in `.\data2\data.mat` or `.\data2\data_time.mat`, please refer to demo files for more details.
- Results are demonstrations of id-based and RLS algorithm performance on human motion prediction by showing prediction error.
- Running time is not recorded, but normally id-based algorithm are slower than RLS algorithm
