using LinearAlgebra
"""
Ac, r = compress_image(A, p)
In:
*A m×nmatrix
* p scalar in (0, 1]
Out:
* Ac a m × n matrix containing a compressed version of A that can be represented using at most (100 * p)% as many bits
required to represent A * r therankof Ac
"""
function compress_image(A, p)

    m, n = size(A)

    u, s, v = svd(A)
    k = Int(floor(p*m*n/(m+n+1)))
    Ac = u[:, 1:k]*Diagonal(s[1:k])*v[:, 1:k]'

    return Ac
end
