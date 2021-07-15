
## Requirements
- conda create -n ddcrnn_env python=3.7   

conda activate ddcrnn_env

- conda install tensorflow=2.0.0 (or up)
- conda install pandas=1.2.5
- conda install pyaml=20.4.0
- conda install numpy=1.20.2 
- conda install scipy=1.6.2
- pip install --upgrade tables


python ddcrnn_train.py --config_filename=data/dcrnn_config.yaml
