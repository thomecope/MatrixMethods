using LinearAlgebra
"""
D = bestdiag(X::Matrix, Y::Matrix)
Solve `\\arg\\min_{D diagonal} \\| X D - Y \\|_F`
In:
* `X`, `Y` `M × N` matrices
Out:
* `D` `N × N Diagonal`
"""
function bestdiag(X::Matrix, Y::Matrix)::Diagonal

    n = size(X)[2]
    d = zeros(ComplexF64, n)
    for i in 1:n
        d[i] = X[:, i] \ Y[:, i]
    end


    return Diagonal(d)
end
