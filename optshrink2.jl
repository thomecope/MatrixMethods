using LinearAlgebra
"""
Xh = optshrink2(Y::AbstractMatrix, r::Int)
Perform rank− r denoising of data matrix Y using the OptShrink method
by Prof. Nadakuditi in this May 2014 IEEE Tr. on Info. Theory paper: http://doi.org/10.1109/TIT.2014.2311661
In:
− Y 2Darraywhere Y=X+noise andgoalistoestimate X − r estimated rank of X
Out:
− Xh rank− r estimate of X using OptShrink weights for SVD components
This version works even if one of the dimensions of Y is large, as long as the other is sufficiently small.

Based on code from Prof. Nadakuditi: https://web.eecs.umich.edu/~rajnrao/optshrink/
"""
function optshrink2(Y::AbstractMatrix, r::Int)

    Util,stil,Vtil = svd(Y)

    sigmahats = stil[1:r]
    Xnoise_est = stil[r+1:end]
    m, n = size(Y)

    k = m-r
    l = n-r
    
    wopt_hats = zeros(r)

    for idx = 1:r
        theta_hats = sqrt(1/estimateDz2(Xnoise_est,sigmahats[idx], l, k))
        Dpz = estimateDpz2(Xnoise_est,sigmahats[idx], l, k)
        wopt_hats[idx] = -2/(theta_hats^2*Dpz)


    end

    Shat = Util[:,1:r]*Diagonal(wopt_hats)*Vtil[:,1:r]';

    return Shat

end

function estimateDz2(s,z, l, k)

    p1 = 1/(k*l)
    p2 = sum(z./(z^2 .- s.^2)) + abs(k-l)/z
    p3 = (sum(z./(z^2 .- s.^2)))
    return p1*p2*p3

end

function estimateDpz2(s,z, l, k)

    p1 = 1/(l*k) * (sum(z./(z^2 .- s.^2)) + abs(k-l)/z)
    p3 = 1/(l*k) * (sum(z./(z^2 .- s.^2)))

    p2 = sum((-2*z^2)./(z^2 .- s.^2).^2 .+ 1 ./(z^2 .- s.^2))
    p4 = sum((-2*z^2)./(z^2 .- s.^2).^2 .+ 1 ./(z^2 .- s.^2)) - abs(k-l)/z^2

    return p1*p2 + p3*p4;

end
