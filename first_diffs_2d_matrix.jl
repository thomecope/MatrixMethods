using SparseArrays
using LinearAlgebra
"""
`A = first_diffs_2d_matrix(m, n)`

In:
− m and n are positive integers

Out:
−`A`is a`2mn × mn`sparse matrix such that`A*X[:]`computes the
first differences down the columns (along x direction)
and across the (along y direction) of the`m × n`matrix`X`
"""

function first_diffs_2d_matrix(m, n)

    dn = spdiagm(0 => -ones(n), 1 => ones(n-1))
    dn[n, 1] = 1

    dm = spdiagm(0 => -ones(m), 1 => ones(m-1))
    dm[m, 1] = 1

    A1 = kron(sparse(I, n, n), dm)
    A2 = kron(dn, sparse(I, m, m))
    A = spzeros(2, 1)
    A = [A1; A2]

    return A

end
