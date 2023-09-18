<img src="logo.png" width="500">

### Ontology guided Variational Autoencoder

OntoVAE is a package that can be used to integrate biological ontologies into latent space and decoder of Variational Autoencoder models. 
This allows direct retrieval of pathway activities from the model.
OntoVAE can also be used to simulate genetic or drug induced perturbations, as demonstrated in our Bioinformatics paper
'Biologically informed variational autoencoders allow predictive modeling of genetic and drug induced perturbations':
https://academic.oup.com/bioinformatics/article/39/6/btad387/7199588

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
Daria Doncevic, Carl Herrmann, Biologically informed variational autoencoders allow predictive modeling of genetic and drug-induced perturbations, Bioinformatics, Volume 39, Issue 6, June 2023, btad387, https://doi.org/10.1093/bioinformatics/btad387
```
