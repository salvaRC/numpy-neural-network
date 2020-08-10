from setuptools import setup

setup(
    name='numpy_neural_network',
    version='0.01',
    packages=['src', 'src.neural_net', 'src.example_applications', 'src.example_applications.MNIST'],
    url='https://github.com/salvaRC/numpy-neural-network',
    license='MIT',
    author='Salva Rühling Cachay',
    author_email='salvaruehling@gmail.com',
    description='Modular neural network from scratch in numpy',
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.8.0"
)
