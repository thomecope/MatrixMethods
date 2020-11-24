using LinearAlgebra

"""
x = lsgd(A, b ; mu=0, x0=zeros(size(A,2)), nIters::Int=200)
Performs gradient descent to solve the least squares problem:
``\\argmin_x 0.5 \\| b − A x \\|_2``
In:
− `A` `m × n` matrix
− `b` vector of length `m`
Option:
− `mu` step size to use, and must satisfy ``0 < mu < 2 / \\sigma_1(A)^2``
to guarantee convergence,
where ``\\sigma_1(A)`` is the first (largest) singular value.
Ch.5 will explain a default value for `mu`
− `x0` is the initial starting vector (of length `n`) to use.
Its default value is all zeros for simplicity.
− `nIters` is the number of iterations to perform (default 200)
Out:
− `x` vector of length `n` containing the approximate LS solution
"""
function lsgd(A, b ; mu::Real=0, x0=zeros(size(A,2)), nIters::Int=200)

    xhat = A \ b
    out = zeros(nIters)
    x = x0 - mu*A'*(A*x0 - b)
    out[1] = log10(norm(xhat - x)/norm(xhat))
    for k = 2:nIters
        xnew = x - mu*A'*(A*x - b)
        # if xnew == x
        #     break
        # end

        x = xnew
        out[k] = log10(norm(xhat - x)/norm(xhat))
    end

return x, out
end
