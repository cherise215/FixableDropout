# FixableDropout
PyTorch Implementation of Dropout layers with fixable masks

To allow the computation with a fixed computational graph in a neural network with dropout layers, we provide fixable Dropout in 1D/2D/3D: `FixableDropout1d`, `FixableDropout2d` and `FixableDropout3d`. One can simply replace `nn.Dropout1d`/`nn.Dropout2d`/`nn.Dropout3d` with them. Intead of directly saving last generated masks, we keep a record of the mask generator, which is parameter efficient. 

## contact:
Chen Chen, Imperial College London. chen.chen15@imperial.ac.uk

Please star our project if you like it and find it useful.
