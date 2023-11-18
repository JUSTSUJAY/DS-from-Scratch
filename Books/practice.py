# import numpy as np
# import random
# x = 2*np.random.rand(100,1)  # 100 rows and 1 column
# y = 4 + 3*x + np.random.randn(100,1) #100 rows and 1 column

# # print(x)
# # print(y)

# # x_b = np.c_[np.ones((100,1)),x] # this line adds 1 at each instance 
# x_b = np.c_[np.array(np.ones(100)),x] # this line adds 1 at each instance 

# # Normal equation - it is used to get the best values of theta vector which minimizes the cost function
# # θ = ((XT).X)^−1.(XT).y
# theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

# # print(x_b)
# print(theta_best)

# import matplotlib.pyplot as plt
# # plt.scatter(x,y)
# # plt.show()

# x_new = np.array([[0],[2]]) # 2d array [[0],[2]]
# x_new_b = np.c_[np.array(np.ones(2)),x_new] # adding 1 to it [[1,0],[1,2]]
# # print(x_new)
# # print(x_new_b)

# y_predict = x_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(x_new,y_predict,"r-")
# plt.plot(x,y,"b.")
# # plt.legend()  
# plt.axis([0,2,0,15])
# plt.show()

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x,y)
# print(model.intercept_,model.coef_)
# print(model.predict(x_new))

# print(np.linalg.pinv(x_b).dot(y))




import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv("E:\ML\BOOKS\housing.csv")
df.head()   
profile = ProfileReport(df)
profile.to_file(output_file = "Food.html")