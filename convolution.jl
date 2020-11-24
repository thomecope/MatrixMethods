
"""
H, y = convolution(h, x)
Compute discrete convolution of the input vectors via
matrix multiplication, returning both the matrix  H  and result  y
In:
− h vector of length K − x vector of length N
Out:
− H M × N convolution matrix defined by h
− y vector of length M containing the discrete convolution of h and x computed using H .
"""
function convolution(h, x)

    N = length(x)
    K = length(h)

    flippedH = reverse(h)

    convL = N+K-1 #number of rows

    tempVec = hcat(vec(zeros(N-1))', flippedH', zeros(N-1)')

    H = zeros(eltype(h), convL, N)

    start_idx = length(tempVec)-N+1
    curr_row = 1

    for i = start_idx:-1:1
        tester = (tempVec[i:i+N-1])'
        H[curr_row, :] = (tempVec[i:i+N-1])'
        curr_row += 1
    end

    y = H*x

    return H, y
end
