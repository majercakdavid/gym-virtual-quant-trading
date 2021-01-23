def conv2d_out_shape(size, kernel_size, stride, layers=1):
    if not isinstance(layers, int) or layers < 1:
        raise AttributeError("Number of layers must be specified by positive number larger than 0")

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*layers
    
    if not isinstance(kernel_size, list):
        raise AttributeError("Kernel size must be either constant integer for all layers or list of integers specifying kernel size at each layer")

    if isinstance(stride, int):
        stride = [stride]*layers
    
    if not isinstance(stride, list):
        raise AttributeError("Stride size must be either constant integer for all layers or list of integers specifying stride at each layer")

    if layers == 1:
        size = (size - (kernel_size[0] - 1) - 1)//stride[0] + 1
    else:
        size = (size - (kernel_size[0] - 1) - 1)//stride[0] + 1
        return conv2d_out_shape(size, kernel_size[1:], stride[1:], layers-1)
    
    return 1 if size < 1 else size