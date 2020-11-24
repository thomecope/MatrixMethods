"""
x = lsngd(A, b ; x0 = zeros(size(A,2)), nIters = 200, mu = 0)
Perform Nesterov−accelerated gradient descent to solve the LS problem \\argmin_x 0.5 \\| A x − b \\|_2
In:
− A m×n matrix
− b vector of length m
Option:
− x0 initial starting vector (of length n ) to use; default 0 vector. − nIters number of iterations to perform; default 200.
− mu step size, must satisfy 0 < \\mu \\leq 1 / \\sigma_1(A)^2
to guarantee convergence, where \\sigma_1(A) is the first (largest) singular value.
Ch.5 will explain a default value for mu .
Out:
 x  vector of length  n  containing the approximate solution
"""
function lsngd(A::AbstractMatrix{<:Number}, b::AbstractVector{<:Number} ; x0::AbstractVector{<:Number} = zeros(eltype(b), size(A,2)),
nIters::Int = 200, mu::Real = 0)

    xhat = A \ b
    xhatN = norm(xhat)
    out = zeros(nIters)

    t0 = 1; z0 = x0;
    tnew = (1+sqrt(1+4*(t0)^2))/2
    xnew = z0 - mu*A'*(A*z0 - b)
    znew = xnew + (t0 - 1)/tnew * (xnew - x0)

    t = tnew
    x = xnew
    z = znew

    out[1] = log10(norm(xhat - x)/xhatN)
    for k = 2:(nIters)

        tnew = (1+sqrt(1+4*(t^2)))/2
        xnew = z - mu*A'*(A*z - b)
        znew = xnew + (t - 1)/tnew * (xnew - x)

        t = tnew; x = xnew; z = znew
        out[k] = log10(norm(xhat - x)/xhatN)
    end

    # return x
    return x, out
end
