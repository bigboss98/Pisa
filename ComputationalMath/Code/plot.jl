using Plots

function plotQ(Q, q, range, fill)
    """
        plotQ is used to represent the contour plot for 
        f(x) = 1/2 * x^T * Q * x + q * x 
        Param:
            Q(Array{Float64, 2}): matrix to use for quadratic part
            q(Array{Float64, 1}): vector to use for linear part 
            range(Array{Float64, 1}): range of values
            fill(Bool): parameter which estabilish whether contour plot will be filled  
    """
    pyplot()
    x = range[1]:0.5:range[2]
    y = range[1]:0.5:range[2]
    f(x, y) = begin
        ((0.5 * [x y] * Q * [x ; y])[1] + q' * [x; y])[1]
        end
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    Z = map(f, X, Y)
    p_1 = contour(x, y, Z, fill=fill)
    display(plot(p_1));
end

Q = [6 -2; -2 6]
q = [10; 5]
plotQ(Q, q, [-10 10], false)