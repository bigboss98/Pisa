using Plots

function plotQ(Q, q, range)
    gr()
    x = range[1]:0.5:range[2]
    y = range[1]:0.5:range[2]
    f(x, y) = begin
        ((0.5 * [x y] * Q * [x ; y])[1] + q' * [x; y])[1]
        end
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    Z = map(f, X, Y)
    p_1 = contour(x, y, Z)
    display(plot(p_1))
end

Q = [6 2; 2 6]
q = [5 ; -5]
plotQ(Q, q, [-10 10])