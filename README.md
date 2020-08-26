# Tutorial for "Advances in machine learning for molecules" 
_Summer school for Machine Learning in Bioinformatics, Moscow_  (https://cs.hse.ru/ssml/).     
John Bradshaw, August 2020

This repo contains the tutorial `ML_for_Molecules.ipynb` for learning about how to use RDKit and performing regression on
molecules. This is the first time anyone else has used this notebook so please give me feedback and tell me about any bugs
(😬, sorry in advance) in the GitHub issues, or create a PR!

## Installation Instructions

The main notebook to run is at `ML_for_Molecules.ipynb`
This notebook can either be run locally or on Colab. You do not require a GPU (I've been running this notebook on my
 ~6 year old laptop fine). This notebook has been designed to be run using Python 3.7 (also works with 3.6),
with the main requirements being PyTorch (I'm using version 1.6) and RDKit, although we also make use of some other packages too. 
Below I describe how to install these packages through Conda if you want to run the notebook locally and how to install them
in Colab if you want to run the notebook on the cloud.

At the end of the first section of the notebook there are a few code cells which import all the Python modules we need 
-- if this works then everything has probably been installed correctly!


### Conda (if you want to run locally)

[Conda](https://docs.conda.io/en/latest/) is a package manager that works on all the popular operating systems.
 If you do not already have it installed (e.g. through Anaconda) you can install it via Miniconda by following
 the insructions [here](https://docs.conda.io/en/latest/miniconda.html) -- it doesn't matter which version of Python
 you pick at this stage. We can then setup the particular environment we need to run this notebook by using the
  [Conda yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
 I put in the root directory of this repo.

Asumming you have Conda installed, clone or download this repo. Navigate inside. Install our environment by:  
1. `conda env create -f conda_ss_moscow_2020.yml`
2. `conda activate ss_moscow_2020`
3. And then finally check it worked correctly by running `conda env list`.

You can then start Jupyter by `jupyter notebook` and select `ML_for_Molecules.ipynb` 


### Colab (if you want to run on the cloud)
[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) allows you to run the notebook on the cloud.
Go to Colab ([https://colab.research.google.com/](https://colab.research.google.com/)). Click on _File -> Upload_ Notebook.
You can then select the `ML_for_Molecules.ipynb` notebook in the GitHub tab after putting in this repo's url or alternatively
you could download and then upload this notebook yourself. I recommend you expand all cells by going _View -> Expand Sections_.
The first section of the notebook will install the packages that Colab does not provide by default.


