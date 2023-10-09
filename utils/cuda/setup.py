from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='stride_conv',
    ext_modules=[
        CUDAExtension('stride_conv', ['strideconv.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)