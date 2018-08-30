# Human-Motion-Prediction

By Yujiao Cheng, Weiye Zhao, Changliu Liu

### Introduction
**Human-Motion-Prediction** is currently implemented with two different algorithms, RLS-PAA and Identifier-based algorithm.

### Requirements: software

0.	MATLAB 2014a or later.

### Requirements: hardware

CPU, Windows 7 or later, MAC OS.

### Fake Data
0.	Run `fake_data_demo.m` to generate arficial motion system training dataset.
    - **Note**: check opts. parameter in demo files for more parameter setting.

### Demo
0.	Run `id_demo.m` to apply identifier-based algorithm on human motion data.
0.	Run `rls_demo.m` to apply RLS-PAA algorithm on human motion data.

### Offline Neural Network Training
0.	`.\offline_train\trainNN.py` to training offline models on human motion data.
0.  human motion dataset should be set manually. Please find Kinect and CMU mocap datasets in `.\data2`.

### Online Adaptation
0. `id_demo.m` and `rls_demo.m` are demos for two online adaptation algorithms. Check both files for more details.
    - **Note**: check opts. parameter in demo files for more parameter setting.
0. Check other scripts in `./lib` for auxiliary function.

### Results
0. The results in terms of prediction error and prediction motion state of two algorithms on four datasets are stored in `./results/..`, please run `plot_err(error, instance number, 'y label', 'x label')` to see the graph for prediction error after loading the error mat.

**Note:** 
- In all the experiments, online adaptation is performed on smoothed human motion data`.\data2\trainX&Y`. Both ready-for-adaptation `trainX` or `trainY` data are stored in `.\data2\data_time.mat` or `.\data2\cmu_data.mat`, which denote Kinect dataset and CMU mocap dataset respectively.
- artificial system data is not stored, but users can run `fake_data_demo.m` to genreate user defined artificial motion data.
- pre-stored offline trained NN initiation parameters for CMU dataset and artificial systems can be found in `.\para\..`, parameters for Kinect dataset can be found in `.\data2`.
- Running time is not recorded, but normally id-based algorithm are slower than RLS algorithm.
