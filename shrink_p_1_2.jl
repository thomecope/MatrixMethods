"""
out = shrink_p_1_2(v, reg::Real)
Compute minimizer of 1/2 |v − x|^2 + reg |x|^p
for p=1/2 when v is real and nonnegative.
In:
*  v  scalar, vector, or array of (real, nonnegative) input values
*  reg  regularization parameter
Out:
*  xh   solution to minimization problem for each element of  v
(same size as  v )
"""
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
