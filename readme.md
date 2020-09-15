The code for the paper "Dynamic Gaussian Mixture based Deep Generative Model ForRobust Forecasting on Sparse Multivariate Time Series" submitted to AAAI 2021

Instructions on how to use the code by using the dataset USHCN as an example:
1. preprocessing the data for forecasting with the following commands in the terminal (--dataset specifies the name of the dataset while -ms specifies the missing ratio for the dataset, e.g. -ms 0.5 means that 50% of the observations are randomly removed from the dataset before the forecasting task):

cd data/
python3 generate_time_series.py --dataset USHCN -ms 0.0



2. run the forecasting program train.py in the main directory. The arguments for running this program are:


--dataset: the name of the dataset
--model: the model name (DGM2 or DGM2_ODE)
-b: mini-batch size
--epochs: epoch count for training
--GPU: flag of using GPU or not, 
--GPUID: ID of the GPU for running train.py
--max_kl: the maximal coefficient for the KL divergence term in the loss function. For USHCN dataset, we can set it as 5.
--use_gate: flag of using the gate function or not
--gaussian: the parameter gamma to balance the dynamic component and the basis mixture component in the dynamic gaussian mixture distribution, which will take effect when --use_gate is not used

Examples:

with GPU (suppose the GPU ID is 0) and gate function:

python3 train.py --dataset USHCN --model DGM2 -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --use_gate


without GPU but using gate function:
python3 train.py --dataset USHCN --model DGM2 -b 100 --epochs 50 --max_kl 5 --use_gate


with GPU (suppose the GPU ID is 0) but without using gate function:

python3 train.py --dataset USHCN --model DGM2 -b 100 --epochs 50 --GPU --GPUID 0 --max_kl 5 --gaussian 0.001


without GPU but using gate function:
python3 train.py --dataset USHCN --model DGM2 -b 100 --epochs 50 --max_kl 5 --gaussian 0.001



