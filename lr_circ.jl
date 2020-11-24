using FFTW
using LinearAlgebra
"""
Ah = lr_circ(A, K::Int)
Compute the rank-at-most-`K` best approximation to circulant matrix `A`
In:
- `A` : `N × N` circulant matrix
- `K` : rank constraint (nonnegative integer: 0, 1, 2, ...)
Out:
- `Ah` : `N × N` best approximation to `A` having rank ≤ `K`
"""
function lr_circ(A, K::Int)::Matrix



    n,_ = size(A)
    evals = (fft(A[:,1])) #get lambdas
    sval = abs.(evals)    #get sigmas
    ssign = sign.(evals)  #get their signs

    K = min(K, length(evals))

    key = sortperm(sval, rev = true) #to keep all K, we want to keep every index ≤ K

    v = zeros(ComplexF64, n,n)

    for i in 1:n
        # if rank_key[i] <= K
        wk = exp(im*2*pi*(i-1)/n)
        v[:, i] = 1/sqrt(n) * wk.^(0:n-1)
             # (vect*vect').*sval[i].*ssign[i]
        # end
    end

    #sorting
    svals_s = sval[key]
    ssign_s = ssign[key]
    v_s = v[:, key]

    return v_s[:, 1:K]*Diagonal(svals_s[1:K].*ssign_s[1:K])*v_s[:,1:K]'
end
