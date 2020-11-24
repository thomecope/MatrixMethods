using LinearAlgebra
"""
tf = inrange(B, z)
Return `true` or `false` depending on whether `z` is in the range of `B`
to within numerical precision. Must be as compute efficient as possible.
In:
* `B` a `M × N` matrix
* `z` vector of length `M`
"""
function inrange(B::AbstractMatrix, z::AbstractVector)

    U,s,V = svd(B)
    r = rank(Diagonal(s))
    Ur = U[:,1:r]

    zhat = Ur*(Ur'*z)
    # out = (isapprox(zhat, z))? true: false

    return (isapprox(zhat, z)) ? true : false
end
