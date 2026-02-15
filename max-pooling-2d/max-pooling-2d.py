def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    H = len(X)
    W = len(X[0])
  
    out_h = H // pool_size
    out_w = W // pool_size

    out = [[float('-inf')] * out_w for _ in range(out_h)]

    for i in range(out_h):
        y_start = i * pool_size
        for j in range(out_w):
            x_start = j * pool_size
            local_max = float('-inf')

            # Loop over the pool_size x pool_size window
            for dy in range(pool_size):
                for dx in range(pool_size):
                    val = X[y_start + dy][x_start + dx]
                    if val > local_max:
                        local_max = val

            out[i][j] = local_max

    return out
    