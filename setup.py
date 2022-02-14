import setuptools

with open("./README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="scAAnet",
  version="1.0.0",
  author="Yuge Wang",
  author_email="wangyuge22@qq.com",
  description="An implementation of nonlinear archetypal analysis on single-cell RNA-seq data through autoencoder",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/AprilYuge/scAAnet_latest",
  install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'tensorflow',
    'sklearn',
    'fastcluster',
    'statsmodels',
    'seaborn',
    'matplotlib',
    'anndata'
  ],
  packages=['scAAnet'],
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)