using Printf
using LinearAlgebra 
using Plots
include("utility.jl")

function SGQ(Q, q, x, fStar=Inf, epsilon=1e-6, max_iter=1000)
    """
    Apply the Steepest Descent algorithm with exact line search to the quadratic function

        f(x) = 1/2 x^T Q x + q x

    Param:
        - Q ([ n x n ] real symmetric matrix, 
               not necessarily positive semidefinite): the Hessian (quadratic part) of f
        - q ([ n x 1 ] real column vector): the linear part of f
        - x ([ n x 1 ] real column vector): the point where to start the algorithm from.
        - fStar (real scalar, optional, default value Inf): optimal value of f.
          if a non-Inf value is provided it is used to print out stasistics about
          the convergence speed of the algorithm
        - epsilon (real scalar, optional, default value 1e-6): the accuracy in the 
          stopping criterion: the algorithm is stopped when the norm of the
          gradient is less than or equal to epsilon
        - max_iter (integer scalar, optional, default value 1000): the maximum
          number of iterations
    
    Output:
        - x ([ n x 1 ] real column vector): either the best solution found so far
          (possibly the optimal one) or a direction proving the problem is
          unbounded below, depending on case
    
        - status (string): a string describing the status of the algorithm at
          termination 
            = 'optimal': the algorithm terminated having proven that x is a(n
              approximately) optimal solution, i.e., the norm of the gradient at x
              is less than the required threshold
    
            = 'unbounded': the algorithm terminated having proven that the problem
               is unbounded below: x contains a direction along which f is
               decreasing to - Inf, either because f is linear along x and the
               directional derivative is not zero, or because x is a direction with
               negative curvature
    
           = 'stopped': the algorithm terminated having exhausted the maximum
             number of iterations: x is the best solution found so far, but not
             necessarily the optimal one
    """
    norm_gradient = Inf
    iteration = 1

    @printf("Gradient Method\n")
    @printf("iter\tf(x)\t\t||nabla f(x)||\t")
    if fStar > -Inf
        @printf("f(x) - f*\t rate")
        previous_function_value = Inf
    end
    @printf("\n\n")

    while (norm_gradient > epsilon && iteration <= max_iter)
        function_value = 0.5 * x' * Q * x + q' * x
        gradient = Q * x + q
        norm_gradient = norm(gradient)
        alpha = norm_gradient^2 / (gradient' * Q * gradient)
        
        @printf("%4d\t%1.8e\t\t%1.4e", iteration, function_value, norm_gradient)
        if fStar > -Inf
            @printf("\t%1.4e", function_value - fStar)
            if previous_function_value < Inf
                @printf( "\t%1.4e" , ( function_value - fStar ) / ( previous_function_value - fStar ) )
            end
            previous_function_value = function_value
        end
        @printf("\n")

        plot_next= (x - alpha * gradient)
        
        display(plot!([(x[1], x[2]), (plot_next[1], plot_next[2])]));

        x = x - alpha * gradient 
        iteration = iteration + 1
        wait_for_key("")
    end
    if norm_gradient < epsilon
      return x, "optimal"
    elseif iteration > max_iter
      return x, "stopped"  
    end
end