# Flattened_CNN
This repository reproduced a paper.
[2014 Flattened convolutional neural networks for feedforward acceleration](http://arxiv.org/abs/1412.5474)

There is a lua and cuda code version
https://github.com/jhjin/flattened-cnn

But it is difficult to understand, so I developed an easy to understand version base on pyTorch.

My version has a disadvantage, too slow. This is because $img2col$ function is implemented by for loop.

In the future, I prepare implement img2col by cuda code.

reference: 
1. https://zhuanlan.zhihu.com/p/40951745

