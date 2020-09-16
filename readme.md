# The code for the paper "Dynamic Gaussian Mixture based Deep Generative Model ForRobust Forecasting on Sparse Multivariate Time Series" submitted to AAAI 2021


## Prerequisites:
install conda, pytorch, matplotlib, pandas, scikit-learn tensorboardX, torchdiffeq (see the instructions in https://github.com/rtqichen/torchdiffeq)



## Instructions on how to use the code to do the forecasting tasks on the USHCN, KDD_CUP and MIMIC-III dataset:
1. preprocessing the data for forecasting with the following commands in the terminal:


To generate data for forecasting on USHCN dataset:

```
cd data/
python3 generate_time_series.py --dataset USHCN
```

To generate data for forecasting on KDDCUP dataset:

```
cd data/
python3 generate_time_series.py --dataset KDDCUP
```



2. run the forecasting program train.py in the main directory. The arguments for running this program are:


--dataset: the name of the dataset (KDDCUP or USHCN)

--model: the model name (DGM2_L or DGM2_O, DGM2_L uses LSTM for transition while DGM2_O uses ODE for transition)

-b: mini-batch size

--epochs: epoch count for training

--GPU: flag of using GPU or not

--GPUID: ID of the GPU for running train.py

--max_kl: the maximal coefficient for the KL divergence term in the loss function. We use annealing technique to tune the coefficient during the training process.

--use_gate: flag of using the gate function or not

--gaussian: the parameter gamma to balance the dynamic component and the basis mixture component in the dynamic gaussian mixture distribution, which will take effect when --use_gate is not used

### Running examples on USHCN dataset:

#### with GPU (suppose the GPU ID is 0) and gate function:



use DGM2_L:
```
python3 train.py --dataset USHCN --model DGM2_L -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --use_gate
```

or

use DGM2_O:
```
python3 train.py --dataset USHCN --model DGM2_O -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --use_gate
```


without GPU but using gate function:

use DGM2_L:

```
python3 train.py --dataset USHCN --model DGM2_L -b 100 --epochs 50 --max_kl 5 --use_gate
```

or

use DHM2_O:

```
python3 train.py --dataset USHCN --model DGM2_O -b 100 --epochs 50 --max_kl 5 --use_gate
```

with GPU (suppose the GPU ID is 0) but without using gate function:


use DGM2_L:
```
python3 train.py --dataset USHCN --model DGM2_L -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --gaussian 0.001
```


or 

use DGM2_O:

```
python3 train.py --dataset USHCN --model DGM2_O -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --gaussian 0.001
```



without GPU but using gate function:

use DGM2_L
```
python3 train.py --dataset USHCN --model DGM2_L -b 100 --epochs 50 --max_kl 5 --gaussian 0.001
```

or 

use DGM2_O:
```
python3 train.py --dataset USHCN --model DGM2_O -b 100 --epochs 50 --max_kl 5 --gaussian 0.001
```



### Running examples on KDDCUP dataset:

with GPU (suppose the GPU ID is 0) and gate function:
```
python3 train.py --dataset KDDCUP --model DGM2 -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --use_gate
```

without GPU but using gate function:
```
python3 train.py --dataset KDDCUP --model DGM2 -b 100 --epochs 50 --max_kl 5 --use_gate
```

with GPU (suppose the GPU ID is 0) but without using gate function:
```
python3 train.py --dataset KDDCUP --model DGM2 -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --gaussian 0.001
```

without GPU but using gate function:
```
python3 train.py --dataset KDDCUP --model DGM2 -b 100 --epochs 50 --max_kl 5 --gaussian 0.001
```



