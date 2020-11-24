using LinearAlgebra
"""
Aa = procrustes(B, A ; center::Bool=true, scale::Bool=true)
In:
`B` and `A` are `d × n` matrices
*
Option:
* *
`center=true/false` : consider centroids? `scale=true/false` : optimize alpha or leave scale as 1?
Your solution needs only to consider the defaults for these.
Out:
`Aa` `d × n` matrix containing `A` Procrustes−aligned to `B`
*
Returns `Aa = alpha * Q * (A − muA) + muB`, where `muB` and `muA` are
the `d × n` matrices whose rows contain copies of the centroids of
`B` and `A`, and `alpha` (scalar) and `Q` (`d × d` orthogonal matrix) are
the solutions to the Procrustes + centering / scaling problem
`\\argmin_{alpha, muA, muB, Q: Q'Q = I} \\| (B − muB) − alpha * Q (A − muA) \\|_F` """
function procrustes(B, A ; center::Bool=true, scale::Bool=true)

    n = size(A,2)
    muA = 1/n * A * ones(n)
    muB = 1/n * B * ones(n)
    A0 = A - muA * ones(n)'
    B0 = B - muB * ones(n)'
    u, _, v = svd(B0*A0')
    Q = u*v'
    alpha = tr(B0*A0'*Q')/tr(A0*A0')
    Aa = alpha * Q * (A-muA*ones(n)') + muB*ones(n)'

    return Aa
end
