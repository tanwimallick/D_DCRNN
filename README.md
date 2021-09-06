# D-DCRNN: Dynamic Diffusion Convolution Recurrent Neural Network for spatio-temporal Forecasting

D-DCRNN is develped based on the diffusion convolutional recurrent neural network(DCRNN), which was originally developed for highway trans-portation  forecasting.  The  key  difference  betweenthe  highway  traffic  DCRNN  and D-DCRNN is  the  way  in which  the  connectivity  between  the  nodes  is  considered  and the  data  that  is  handled.  While  we  see  that  traffic  network sseem  to  have  regular  seasonality  patterns  in  terms  of  time of the day   and   week days,   WAN network   traffic   has   more   pseudo-random patterns, where the flows depend on users connectivity and  active  projects.  Additionally,  in  DCRNN  approach,  the connectivity  is  static  and  computed  on  the  basis  of  (driving) distance;  whereas  in D-DCRNN,  the  connectivity  is  dynamicand computed on the basis of the current state of the networkflow traffic. This approach is designed to explicitly model the dynamic nature of the WAN traffic. 

## Step-by-Step installation instructions 
Download the repo 

$ cd D_DCRNN

## Requirements
- conda create -n ddcrnn_env python=3.7   

conda activate ddcrnn_env

- conda install tensorflow=2.4.0
- conda install pandas=1.2.5
- conda install pyaml=20.4.0
- conda install numpy=1.19.5
- conda install scipy=1.6.2
- pip install --upgrade tables

## Instruction to run the code
$ cd ddcrnn2.0

$ python ddcrnn_train.py --config_filename=data/dcrnn_config.yaml

The model generates prediction of D-DCRNN.

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@article{mallick2020dynamic,
  title={Dynamic graph neural network for traffic forecasting in wide area networks},
  author={Mallick, Tanwi and Kiran, Mariam and Mohammed, Bashir and Balaprakash, Prasanna},
  journal={arXiv preprint arXiv:2008.12767},
  year={2020}
}
```

