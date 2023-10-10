# QuadTreeVision
Traditional Convolution Neural Network works by processing every single section of an Image in a fine-to-coarse manner. Resulting in many redundancy, sparse Convolution layers have been proposed to tackle such computational redundancy. Human vision works by selecting important parts and pays attention (hard attention which allocate computational load). This repo attempts to build a learnable hard attention, which only process sparse signals from Image, while trying to achieve similar performance as the Convolutional baseline models.

QuadTree Node can be used to compress an Image in a coarse-to-fine manner
![image](https://github.com/fangyuan-ksgk/QuadTreeVision/assets/66006349/80910730-1931-48b4-95c3-8ca575841f12)
