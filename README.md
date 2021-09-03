# D-DCRNN: Dynamic Diffusion Convolution Recurrent Neural Network for spatio-temporal Forecasting

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
