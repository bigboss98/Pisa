import numpy as np
import util

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    logistic_classifier = LogisticRegression()
    logistic_classifier.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    y_valid = logistic_classifier.predict(x_valid)
    
    util.plot(x_valid, y_valid, logistic_classifier.theta, "logistic_regression.jpg")

    #np.savetxt(save_path)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        
        self.theta = np.zeros(x.shape[1])
        error = 1#default value of 1 to avoid stop on first iteration of Newton

        while error > self.eps:
            prec_theta = self.theta

            gradient = np.array([1 / x.shape[1] * 
                                np.sum([(self.hypothesis_function(x_val) - y_val) * x_val[j]
                                                        for x_val in x for y_val in y])
                                for j in range(len(self.theta))])
            
            self.theta = prec_theta - (self.step_size / x.shape[1]) *  gradient 
            error = np.linalg.norm([self.theta - prec_theta])
            print("After iteration we have error: ", error)

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return np.array([np.round(self.hypothesis_function(x_val)) for x_val in x])

    def hypothesis_function(self, x):
        """
            Return value of h(x) = g(\theta^T * x) = 1 / 1 + e ^{-\theta^T * x}

            Args:
                x: a Input value

            Returns:
                Output the hypothesis value h(x)
        """
        return 1 / (1 + np.exp(-np.dot(self.theta, x)))  

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
