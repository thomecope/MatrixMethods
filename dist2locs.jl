using LinearAlgebra
"""
Xr = dist2locs(D, d)

In:
* `D` is an `n x n` matrix such that `D[i, j]` is the distance from object `i` to object `j`
* `d` is the desired embedding dimension.

Out:
* `Xr` is an `d x n` matrix whose columns contains the relative coordinates of the `n` objects

Note: MDS is only unique up to rotation and translation,
so we enforce the following conventions on Xr in this order:
* [ORDER] `Xr[i,:]` corresponds to ith largest eigenpair of `C' * C`
* [CENTER] The centroid of the coordinates is zero
* [SIGN] The largest magnitude element of `Xr[i,:]` is positive
"""
function dist2locs(D, d)

    J = size(D,1)

    S = D.^2
    S = 0.5 * (S + S')

    P_orth = I - (1/J)*ones(J)*ones(J)'
    G = -1/2 * P_orth * S * P_orth

    u, s, v = svd(G)
    Xr = Diagonal(sqrt.(s[1:d]))*(v[:, 1:d])'

    for i in 1:d
        _, idx = findmax(abs.(Xr[i,:]))
        a = sign(Xr[i, idx])
        Xr[i,:] = a.*Xr[i,:]
    end

    return Xr
end
