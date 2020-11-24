using LinearAlgebra

"""
    labels = classify_image(test, train, K::Int)

Classify `test` signals using `K`-dimensional subspaces
found from `train`ing data via SVD

In:
* `test` `n x p` matrix whose columns are vectorized test images to be classified
* `train` `n x m x 10` array containing `m` training images for each digit 0-9 (in ascending order)
* `K` in `[1, min(n, m)]` is the number of singular vectors to use during classification

Out:
`labels` vector of length `p` containing the classified digits (0-9) for each test image
"""
function classify_image(test, train, K::Int)
    n = size(test)[1] # n is the dimension of our vector (the number of pixels in each image)
    p = size(test)[2] # p is the number of test vectors
    m = size(train)[2] # m is the number of training samples we have per digits
    d = size(train)[3]; # d is the number of digits

    #Making allocations
    Q = zeros((n, K, d))
    err = zeros((d, p))

    #Finding LRA
    for i in 1:d
        u, _, _ = svd(train[:,:,i])
        Q[:, :, i] = u[:, 1:K]
    end


    #Computing error
    for i in 1:d
        err[i,:] = sum((test-Q[:,:,i]*(Q[:,:,i]'*test)).^2, dims=1)
    end

    #Assigning labels
    _, prelabels = findmin(err, dims=1)
    labels = [ i[1] for i in prelabels ] .- 1


end
