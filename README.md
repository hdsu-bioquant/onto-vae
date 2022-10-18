# OntoVAE

OntoVAE is a package that can be used to integrate biological ontologies into latent space and decoder of Variational Autoencoder models. 
This allows direct retrieval of pathway activities from the model.
OntoVAE can also be used to simulate genetic or drug induced perturbations, as demonstrated in our manuscript 
'Biologically informed variational autoencoders allow predictive modeling of genetic and drug induced perturbations':
https://www.biorxiv.org/content/10.1101/2022.09.20.508703v2

## Installation

```
conda create -n ontovae python=3.7
conda activate ontovae
pip install onto-vae
```

## Usage

In python, import neccessary modules

```
from onto_vae.ontobj import *
from onto_vae.vae_model import *
```

For on example on how to use our package, you can check out the vignette! If you want to run the vignette as jupyter notebook, inside your conda environment, also install jupyter and then open the jupyter notebook:

```
conda install jupyter
jupyter notebook
```

Preprocessed ontobj for Gene Ontology (GO) and Human Phenotype Ontology (HPO) and pretrained models are available under:
https://figshare.com/projects/OntoVAE_Ontology_guided_VAE_manuscript/146727

## Citation

If you use OntoVAE for your research, please cite:
```
Doncevic, D., Herrmann, C. Biologically informed variational autoencoders allow predictive modeling of genetic and drug induced perturbations. bioRxiv (2022) doi: https://doi.org/10.1101/2022.09.20.508703 
```
