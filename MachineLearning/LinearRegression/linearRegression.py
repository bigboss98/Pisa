import matplotlib.pyplot as plt
import numpy as np


def load_data(file_name):
    data = np.loadtxt("ex1data.txt", delimiter=",")
    values = [(elem[0], elem[1]) for elem in data]
    return values

def visualization(data):
    xvalues = [elem[0] for elem in data]
    yvalues = [elem[1] for elem in data]
    plt.scatter(xvalues, yvalues, vmin=[-5, 0], vmax=[5, 100], marker="x")
    plt.xlabel("Population of City in 10000s")
    plt.ylabel("Profit in $10000s")
    plt.show()

class LinearRegression():
    """
        Class to implement a Linear Regresssion 
    """

    def __init__(self, data):
        """
            Linear Regression constructor 
            :param self, current object
            :param data, array of (x, y) point used as dataset points
        """
        self.xvalues = [(1, elem[0]) for elem in data]
        self.yvalues = [elem[1] for elem in data]
        self.theta = np.zeros((2, 1))
        self.methods = ["gradient_descent", "newton", "normal_equation"]

    def cost_function(self):
        return 1 / 2 * len(self.xvalues) * np.var([self.linear_regression(x_val) - y_val for x_val in self.xvalues for y_val in self.yvalues])

    def linear_regression(self, data):
        x_values = np.array(data, ndmin=2)
        return np.inner(np.transpose(self.theta), x_values)

    def fit(self, iterations=1500, alpha=0.01, method="gradient_descent"):
        if method in self.methods:
            eval("self." + method + "(" + str(iterations) + ", " + str(alpha) + ")")        
        else:
            print("Method chosen is not support in Linear Regression")
            
    def predict(self, data):
        data = (1, data)
        return linear_regression()

    
    def gradient_descent(self, iterations, alpha):
        num_iteration = 0
        
        while num_iteration < iterations:
            for j, theta_j in enumerate(self.theta):
                
                theta_j = (theta_j - (alpha / len(self.xvalues))
                                   * np.sum([(self.linear_regression(x_val) - y_val) * x_val for x_val in self.xvalues for y_val in self.yvalues]))

            num_iteration = num_iteration + 1 
            print("Done Iteration: ", num_iteration)


#def normal_equation(data):


#def stocastic_gradient_descent(data):


#def newton(data):

data_values = load_data("ex1data.txt")
visualization(data_values)
linear_regression = LinearRegression(data_values)
print("Cost function before Theta: ", linear_regression.cost_function())
linear_regression.fit()

