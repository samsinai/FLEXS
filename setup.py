import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="flexs",
    version="0.2.8",
    description=(
        "FLEXS: an open simulation environment for developing and comparing "
        "model-guided biological sequence design algorithms."
    ),
    url="https://github.com/samsinai/FLEXS",
    author="Stewart Slocum",
    author_email="slocumstewy@gmail.com",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "cma",
        "editdistance",
        "numpy>=1.16",
        "pandas>=0.23",
        "torch>=0.4",
        "scikit-learn>=0.20",
        "tape-proteins",
        "tensorflow>=2",
        "tf-agents>=0.7.1",
    ],
    include_package_data=True,
    package_data={
        "": [
            "landscapes/data/additive_aav_packaging/*",
            "landscapes/data/rosetta/*",
            "landscapes/data/tf_binding/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
