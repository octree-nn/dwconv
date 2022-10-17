import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dir = './csrc'
sources = ['{}/{}'.format(dir, src) for src in os.listdir(dir)
           if src.endswith('.cpp') or src.endswith('.cu')]

setup(
    name='dwconv',
    version='1.0.0',
    packages=['dwconv'],
    install_requires=["torch"],
    include_package_data=True,
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            name='dwconv.core',
            sources=sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
