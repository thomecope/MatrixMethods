using LinearAlgebra
"""
T = nearptf(X)
Find nearest (in Frobenius norm sense) Parseval tight frame to matrix `X`.
In:
* `X` : `N × M` matrix
Out:
* `T` `N × M` matrix that is the nearest Parseval tight frame to `X`
"""
function nearptf(X::AbstractMatrix)
    u, s, v = svd(X);
    return u*v'
end
