using LinearAlgebra

"""
e = eigtiki(X, beta)
Return the eigenvalues in ascending order of `(X'X + beta I)^+` for `β ≥ 0`,
without calling (or duplicating) functions listed in the problem statement.
The output should be a Vector.
"""
function eigtiki(X::AbstractMatrix, beta::Real)::Vector

    if X == zeros(size(X))
        if beta == 0
            return zeros(size(X)[2])
        else
            return ones(size(X)[2]) .* 1/beta
        end
    end

    vect = svdvals(X'X + beta*I)
    r = rank(Diagonal(vect))
    num_zeros = length(vect)-r

    # out = [1 ./vec[1:r]; zeros(size(X[2])-r)

    return [vect[r+1:end]; 1 ./vect[1:r]]
end
