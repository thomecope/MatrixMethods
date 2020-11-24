using LinearAlgebra
"""
Xh = optshrink1(Y::AbstractMatrix, r::Int)
Perform rank−r denoising of data matrix Y using the OptShrink method by Prof. Nadakuditi in this May 2014 IEEE Tr. on Info. Theory paper:
http://doi.org/10.1109/TIT.2014.2311661
In:
− Y 2Darraywhere Y=X+noise and goal is to estimate X
− r estimated rank of X
Out:
− Xh rank− r estimate of X using OptShrink weights for SVD components
This version works only if the size of  Y  is sufficiently small,
because it performs calculations involving arrays roughly of
size(Y'*Y) and size(Y*Y') , so neither dimension of Y can be large.

Based on code from Prof. Nadakuditi: https://web.eecs.umich.edu/~rajnrao/optshrink/
"""
function optshrink1(Y::AbstractMatrix, r::Int)

    m, n = size(Y)
    mins = min(m, n)
    Util,stil,Vtil = svd(Y)
    temp = Matrix(Diagonal(stil))
    Stil = zeros(size(Y))
    Stil[1:mins, 1:mins] = temp
    # @show Stil

    sigmahats = stil[1:r]
    # @show sigmahats
    Xnoise_est = Stil[(r+1):end,(r+1):end]
    # @show Xnoise_est
    wopt_hats = zeros(r)

    for idx = 1:r
        theta_hats = sqrt(1/estimateDz(Xnoise_est,sigmahats[idx]))
        # @show estimateDz(Xnoise_est,sigmahats[idx])
        # @show theta_hats
        Dpz = estimateDpz(Xnoise_est,sigmahats[idx])
        # @show Dpz
        wopt_hats[idx] = -2/(theta_hats^2*Dpz)
        # @show wopt_hats
    end

    Shat = Util[:,1:r]*Diagonal(wopt_hats)*(Vtil[:,1:r])';

    return Shat

end

function estimateDz(X,z)
    n,m = size(X);
    # return D1z*D2z
    # @show X
    z2IXXt = z^2*I - X*X';
    z2IXtX = z^2*I - X'*X;
    invz2XtX = inv(z2IXtX);
    # @show invz2XtX
    invz2XXt = inv(z2IXXt);
    # @show invz2XXt

    @show 1/(m*n)
    D1z = tr(z*invz2XXt);
    @show D1z
    D2z = tr(z*invz2XtX);
    @show D2z

    Dz = 1/(m*n)*D1z*D2z;
    # @show Dz
    return Dz

end

function estimateDpz(X,z)

    n,m = size(X);


    z2IXXt = z^2*I - X*X';
    z2IXtX = z^2*I - X'*X;
    invz2XtX = inv(z2IXtX);
    invz2XXt = inv(z2IXXt);

    D1z = 1/n*tr(z*invz2XXt);
    D2z = 1/m*tr(z*invz2XtX);


    D1zp = 1/n*tr(-2*z^2*invz2XXt^2+invz2XXt);
    D2zp = 1/m*tr(-2*z^2*invz2XtX^2+invz2XtX);

    Dpz = D1z*D2zp+D1zp*D2z;

    return Dpz;
end
