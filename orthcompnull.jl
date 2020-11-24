using LinearAlgebra
"""
R = orthcompnull(A, X)
Project each column of X onto the orthogonal complement of the null space of the input matrix A .
In:
* A M×N matrix
* X vector of length N , or matrix with N rows and many columns
Out:
*  R  : vector or matrix of size ??? (you determine this)
For full credit, your solution should be computationally efficient!
"""
function orthcompnull(A, X)
    _, _, v = svd(A)
    # vec trick : vec(AXB) = kron(transpose(B), A)*vec(X)
    # normally I would do v*v'*x, so A = v, X = v', B = x
    return reshape(kron(transpose(X), v)*vec(v'), size(X))
end
