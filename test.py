# import torch
# import torch.nn as nn
# from Sparsemax.Sparsemax import Sparsemax
# 
# sparsemax = Sparsemax(dim=1)
# softmax = torch.nn.Softmax(dim=1)
# 
# def create_net(n_inputs, n_outputs, n_layers = 1, 
#     n_units = 100, nonlinear = nn.Tanh, add_softmax = False):
#     layers = [nn.Linear(n_inputs, n_units)]
#     for i in range(n_layers):
#         layers.append(nonlinear())
#         layers.append(nn.Linear(n_units, n_units))
# 
#     layers.append(nonlinear())
#     layers.append(nn.Linear(n_units, n_outputs))
#     if add_softmax:
#         layers.append(nn.Softmax(dim=-1))
#     
#     return nn.Sequential(*layers)
# 
# x = torch.rand((400, 10))
# 
# model = create_net(10, 10, add_softmax = True)
# 
# y = model(x)
# 
# print(torch.sum(y, -1))
# 
# 
# 
# 
# logits = torch.randn(2, 5)
# print("\nLogits")
# print(logits)
# 
# softmax_probs = softmax(logits)
# print("\nSoftmax probabilities")
# print(softmax_probs)
# 
# 
# softmax_probs2 = softmax(torch.log(softmax_probs))
# 
# print('here::', torch.norm(softmax_probs2 - softmax_probs))
# 
# 
# 
# sparsemax_probs = sparsemax(torch.log(softmax_probs))
# print("\nSparsemax probabilities")
# print(sparsemax_probs)
# 
# sparsemax_probs = sparsemax(logits)
# print("\nSparsemax probabilities")
# print(sparsemax_probs)




# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# sns.set(style="ticks")
# tips = sns.load_dataset("tips")
# print(tips)
# 
# 
# # fig, ax1 = plt.subplots(figsize=(10,10))
# # g = sns.FacetGrid(tips, col="time")
# 
# titles = {'Male':'1', 'Female':'2'}
# 
# g = sns.FacetGrid(tips, col="sex")
# g.map(plt.plot, "total_bill", "tip", alpha=.7)
# g.add_legend();
# 
# axes = g.axes.flatten()
# axes[0].set_title("1")
# axes[1].set_title("2")
# 
# plt.show()


import numpy as np

from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


X = np.random.random((1000, 10))
a = np.random.random((10, 3))
y = np.dot(X, a) + np.random.normal(0, 1e-3, (1000, 3))

# fitting
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear')).fit(X, y)

# predicting
print(np.mean((multioutputregressor.predict(X) - y)**2, axis=0))  # 0.004, 0.003, 0.005












