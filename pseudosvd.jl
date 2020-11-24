using LinearAlgebra
"""
(Ur,Sr,Vr) = pseudosvd(A)
Return 3 matrices in a compact SVD of the Moore-Penrose pseudo-inverse of `A`
without calling (or duplicating) Julia's built-in pseudo-inverse function.
The returned triplet of matrices should satisfy `Ur*Sr*Vr' = A^+`
to within appropriate numerical precision.
"""
function pseudosvd(A::AbstractMatrix)
    U,s,V = svd(A)
    r = rank(Diagonal(s))
    Vr = reverse(U[:,1:r], dims = 2)
    Ur = reverse(V[:,1:r], dims = 2)
    sr = Diagonal(reverse((1 ./ s[1:r])))

    return Ur, sr, Vr
end
