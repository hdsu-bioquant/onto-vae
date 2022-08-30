# OntoVAE
OntoVAE is a package that can be used to integrate biological ontologies into latent space and decoder of Variational Autoencoder models. 
This allows direct retrieval of pathway activities from the model.
OntoVAE can also be used to simulate genetic or drug induced perturbations, as demonstrated in our manuscript 
'Biologically informed variational autoencoders allow predictive modeling of genetic and drug induced perturbations'.

Mainly, the workflow for training OntoVAE models consists of two steps.

## 1. Creation of an ontology object

For this, we use the Ontobj() class from onto-vae package. Ontobj() requires two input files:

1. a biological ontology in obo format
2. a tab separated text file containing the mapping from genes to ontology terms, e.g.
  ``` 
  CLPP	HP:0000252
  CLPP	HP:0004322  
  CLPP	HP:0001250
  ```
  
We can then create our Ontobj as follows
  ```
  from onto_vae.ontobj import *
  import pickle
  
  hpo = Ontobj(description='HPO')  # can be anything but should contain info about ontology
  ```
Next, we will initialize our ontology object with the following command. 
  
  ```
  hpo.initialize_dag(obo='/path/to/HPO/data/hp.obo',
                     gene_annot='/path/to/HPO/data/gene_term_mapping.txt')
  ```

Then we will trim the ontology with user defined thresholds (please see our manuscript for more detailed explanation).

  ```
  hpo.trim_dag(top_thresh=1000, 
               bottom_thresh=10)
  ```
And finally, create masks that can be used by the VAE model.
  ```
  hpo.create_masks(top_thresh=1000,
                   bottom_thresh=10)
  ```
  
Now, we can store datasets in our object that will be matched to the genes of the ontology. 
The dataset should be provided as h5ad file, 
comma-separated csv file (genes in rows, samples in columns),
tab-separated text file (genes in rows, samples in columns), or
pandas dataframe (genes in index, samples in columns)
The name argument is used so that multiple datasets can be stored and distinguished.
```
hpo.match_dataset(expr_path = '/path/to/dataset/train_pbmc.h5ad',
                  name='Kang_PBMC')
```
Finally, we want to store our Ontobj, so that we can use it for model training.

```
with open('/path/to/HPO/data/HPO_ontoobj.pickle', 'wb') as f:
    pickle.dump(hpo, f) 
```

## 2. Training of OntoVAE model

To train an OntoVAE model, we need to import an Ontobj first.

```
import pickle
import torch

with open('/path/to/HPO/data/HPO_ontoobj.pickle', 'rb') as f:
    hpo = pickle.load(f) 
```

Then we can initialize a new OntoVAE model as follows and move it to the GPU:
```
hpo_model = OntoVAE(ontobj=hpo,              # the Ontobj we will use
                    dataset='Kang_PBMC',     # which dataset from the Ontobj to use for model training
                    top_thresh=1000,         # which trimmed version to use
                    bottom_thresh=10,        # which trimmed version to use
                    neuronnum=3)             # number of neurons per term
                    
hpo_model.to(model.device)
```

To train the model, we call the following function:
```
hpo_model.train_model(modelpath='/path/to/models/best_model.py',     # where to store the best model
                     lr=1e-4,                                        # the learning rate
                     kl_coeff=1e-4,                                  # the weighting coefficient for the Kullback Leibler loss
                     batch_size=128,                                 # the size of the minibatches
                     epochs=300,                                     # over how many epochs to train
                     log=False)                                      # if the run should be logged to Neptune
```
After training, we can load the best model to retrieve pathway activities.
```
checkpoint = torch.load('/path/to/models/best_model.py')
hpo_model.load_state_dict(checkpoint['model_state_dict'])

hpo_act = hpo_model.get_pathway_activities(ontobj=hpo,
                                           dataset='Kang_PBMC')
```
