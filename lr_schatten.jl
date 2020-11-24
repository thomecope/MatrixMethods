using LinearAlgebra
"""
lr_schatten(Y, reg::Real)
Compute the regularized low−rank matrix approximation as the minimizer over X
of 1/2 \\|Y − X\\|^2 + reg R(x)
where R(X) is the Schatten p−norm of X raised to the pth power, for p=1/2 ,
i.e., R(X) = \\sum_k (\\sigma_k(X))^{1/2}
In:
− Y MbyN matrix
− reg regularization parameter
Out:
− Xh M by N solution to above minimization problem
"""
function lr_schatten(Y, reg::Real)
    
    ur, s, vr = svd(Y)
    sr = shrink_p_1_2(s, reg)
    return ur*Diagonal(sr)*vr'

end

function shrink_p_1_2(v, reg::Real)

    xhat = zeros(size(v))
    for i in 1:length(v)
        if v[i] > (3/2 * reg^(2/3))
            xhat[i] = 4/3*v[i]*cos(1/3*acos(-3^(3/2)*reg/(4*v[i]^(3/2))))^2

        else
            xhat[i] = 0
        end
    end


    return xhat

end
