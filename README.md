# simple-cnn
This is a simple CNN feedforward network. I extract some important layers from caffe. Armadillo is utilized to speed up matrix operations. The convolution operation is converted into matrix multiplication like caffe.

Details:

![](https://github.com/goodluckcwl/simple-cnn/raw/master/images/1.png)
![](https://github.com/goodluckcwl/simple-cnn/raw/master/images/2.png)

# Performance
I have not tested the performance quantificationally. Qualitatively speaking, it takes about 20-30 ms to run a feedforward on a small CNN network(10 conv layers, input size:55*47) on i5-4590 without GPU.

# Dependence
- Armadillo
- Opencv
