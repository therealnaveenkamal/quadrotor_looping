from setuptools import setup, find_packages

setup(
    name="quadrotor_looping",
    version="1.0",
    description="A quadrotor simulation package with SQP-based control",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "qpsolvers",
        "qpsolvers[cvxopt]",
        "scipy",
        
    ],
    entry_points={
        "console_scripts": [
            "run_simulation=quadrotor_looping.main:main",
        ],
    },
)
