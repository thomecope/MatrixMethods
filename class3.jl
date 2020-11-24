using LinearAlgebra
"""
For exam: class3(U,x)
"""
function class3(U, x)
    alpha = zeros(size(U, 2))
    for i in 1:size(U,2)
        # @show (U[:,i])'
        # @show x
        mat = (U[:,i])'*x
        mat = mat[1]

        alpha[i] = norm(x - mat*U[:,i])
        # alpha[i] = mat*U[:,i]
    end

    _, index = findmin(alpha)

    return index
end
