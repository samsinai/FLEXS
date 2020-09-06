import setuptools

with open("README.md") as f:
    long_name = f.read()

setuptools.setup(
    name="flexs",
    version="0.2.0",
    description="FLEXS: an open simulation environment for developing and comparing model-guided biological sequence design algorithms.",
    url="https://github.com/samsinai/FLSD-Sandbox",
    author="",
    author_email="",
    license="Apache",
    long_name=long_name,
    long_name_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
