# OntoVAE
OntoVAE is a package that can be used to integrate biological ontologies into latent space and decoder of Variational Autoencoder models. 
This allows direct retrieval of pathway activities from the model.
OntoVAE can also be used to simulate genetic or drug induced perturbations, as demonstrated in our manuscript 
'Biologically informed variational autoencoders allow predictive modeling of genetic and drug induced perturbations'.

## Installation

In the future, installation via pip will be supported. For now, you can install the package through github as follows:

```
git clone https://github.com/hdsu-bioquant/onto-vae.git
cd onto-vae
```
It is best to first create a new environment, e.g. with conda, and then install the package inside.

```
conda create -n ontovae python=3.7
conda activate ontovae
pip install -r requirements.txt
```

For on example on how to use our package, please see the Vignette! If you want to run the Vignette as Jupyter notebook, inside your conda environment, also install Jupyter and then open the jupyter notebook:

```
conda install jupyter
jupyter notebook
```
