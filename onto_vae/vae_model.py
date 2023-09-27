#!/usr/bin/env python3

import sys
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .modules import Encoder, Decoder, OntoEncoder, OntoDecoder, simple_classifier
from .fast_data_loader import FastTensorDataLoader

###-------------------------------------------------------------###
##                  VAE WITH ONTOLOGY IN DECODER                 ##
###-------------------------------------------------------------###

class OntoVAE(nn.Module):
    """
    This class combines a normal encoder with an ontology structured decoder

    Parameters
    ----------
    ontobj
        instance of the class Ontobj(), containing a preprocessed ontology and the training data
    dataset
        which dataset from Ontobj to use for training
    top_thresh
        top threshold to tell which trimmed ontology to use
    bottom_thresh
        bottom_threshold to tell which trimmed ontology to use
    neuronnum
        number of neurons per term
    drop
        dropout rate, default is 0.2
    z_drop
        dropout rate for latent space, default is 0.5
    labels
        labels of samples if classifier is used
    batches
        batch information of samples if available
    modelpath
        path with pretrained model if pretrained model should be passed
    """

    def __init__(
        self, 
        ontobj, 
        dataset, 
        top_thresh=1000, 
        bottom_thresh=30, 
        neuronnum=3, 
        drop=0.2, 
        z_drop=0.5,
        labels=None,
        batches = None,
        modelpath = None
        ):

        super(OntoVAE, self).__init__()

        if not str(top_thresh) + '_' + str(bottom_thresh) in ontobj.genes.keys():
            raise ValueError('Available trimming thresholds are: ' + ', '.join(list(ontobj.genes.keys())))

        self.ontology = ontobj.description
        self.top = top_thresh
        self.bottom = bottom_thresh
        self.genes = ontobj.genes[str(top_thresh) + '_' + str(bottom_thresh)]
        self.in_features = len(self.genes)
        self.mask_list = ontobj.masks[str(top_thresh) + '_' + str(bottom_thresh)]['decoder']
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.latent_dim = self.layer_dims_dec[0] * neuronnum
        self.layer_dims_enc = [self.latent_dim]
        self.neuronnum = neuronnum
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = labels
        self.batches = batches
        self.one_hot = None
        self.VAE_trained = False
        self.clf_trained = False
        self.modelpath = modelpath

        if not dataset in ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys():
            raise ValueError('Available datasets are: ' + ', '.join(list(ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys())))

        self.X = ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)][dataset]
        
        # set labels (or dummies) for classification
        if self.labels is not None:
            self.y = self.labels
        else:
            self.y = np.zeros(self.X.shape[0])
        self.n_classes = len(set(self.y))

        # set batches (or dummies) for batch correction
        if self.batches is not None:
            self.n_batch = len(set(self.batches))
            self.one_hot = torch.eye(self.n_batch).to(self.device)
            self.batch_info = self.batches
        else:
            self.n_batch = 0
            self.batch_info = np.zeros(self.X.shape[0])

        # parse model information if pretrained model was passed
        if self.modelpath is not None:
            checkpoint = torch.load(modelpath,
                        map_location = torch.device(self.device))
            self.VAE_trained = True
            self.val_loss_min = checkpoint['loss']
            if checkpoint['classifier_trained']:
                self.clf_trained = True
                self.n_classes = checkpoint['n_classes']
            if checkpoint['batch_corrected']:
                self.n_batch = checkpoint['n_batch']
                self.one_hot = torch.eye(self.n_batch).to(self.device)

        # Encoder
        self.encoder = Encoder(self.in_features + self.n_batch,
                                self.latent_dim,
                                self.layer_dims_enc,
                                self.drop,
                                self.z_drop)

        # Decoder
        self.decoder = OntoDecoder(self.in_features,
                                    self.layer_dims_dec,
                                    self.mask_list,
                                    self.latent_dim,
                                    self.n_batch,
                                    self.one_hot,
                                    self.neuronnum)
        
        # Classifier (optional)
        if self.labels is not None:
            self.classifier = simple_classifier(in_features=np.sum(self.layer_dims_dec[:-1]),
                                             n_classes=self.n_classes)

        # load parameters if pretrained model was passed
        if self.modelpath is not None:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)   

    def reparameterize(self, mu, log_var):
        """
        Performs the reparameterization trick.

        Parameters
        ----------
        mu
            mean from the encoder's latent space
        log_var
            log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x, batch=None):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            dataset of which embedding should be generated
        """
        # attach batch information
        if batch is not None:
            x = torch.hstack((x, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            x = torch.hstack((x, torch.zeros((x.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # set to eval mode
        self.eval()

        # encoding
        mu, log_var = self.encoder(x)

        # sample from latent space
        embedding = self.reparameterize(mu, log_var)
        return embedding

    def forward(self, x, batch=None):

        # attach batch information
        if batch is not None:
            x = torch.hstack((x, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            x = torch.hstack((x, torch.zeros((x.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # encode
        mu, log_var = self.encoder(x)
            
        # sample from latent space
        z = self.reparameterize(mu, log_var)
        
        # attach hooks for classification
        if self.labels is not None:
            activation = {}
            hooks = {}
            self._attach_hooks(activation=activation, hooks=hooks)

        # decoding
        reconstruction = self.decoder(z, batch)
        
        if self.labels is not None:
            act = torch.cat(list(activation.values()), dim=1)
            act = torch.hstack((z,act))
            act = torch.stack(list(torch.split(act, self.neuronnum, dim=1)), dim=0).mean(dim=2).T
            output = self.classifier(act)
            for h in hooks:
                hooks[h].remove()
            return reconstruction, mu, log_var, output
        else:
            return reconstruction, mu, log_var


    def get_classification(self, ontobj, dataset, batch=None):
        """
        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for pathway activity retrieval
        """
        if self.ontology != ontobj.description:
            sys.exit('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor 
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # attach batch information
        if batch is not None:
            data = torch.hstack((data, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            data = torch.hstack((data, torch.zeros((data.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # set to eval mode
        self.eval()

        with torch.no_grad():
            # encoding
            mu, log_var = self.encoder(data)

            # sample from latent space
            z = self.reparameterize(mu, log_var)
        
        # attach hooks for classification
        activation = {}
        hooks = {}
        self._attach_hooks(activation=activation, hooks=hooks)

        # decoding
        with torch.no_grad():
            reconstruction = self.decoder(z, batch)
        
        # get activation values
        act = torch.cat(list(activation.values()), dim=1)
        act = torch.hstack((z,act))
        act = torch.stack(list(torch.split(act, self.neuronnum, dim=1)), dim=0).mean(dim=2).T

        # perform classification
        with torch.no_grad():
            output = self.classifier(act)
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        output = output.to('cpu').detach().numpy()
        y_pred = np.argmax(output,axis=1)

        return y_pred

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, mode='val', run=None):
        """
        Parameters
        ----------
        mode
            'train' or 'val' (default): for logging
        run
            Neptune run if training is to be logged
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def classify_loss(self, class_output, y, mode='val', run=None):
        """
        Parameters
        ----------
        mode
            'train' or 'val' (default): for logging
        run
            Neptune run if training is to be logged
        """
        class_loss = nn.CrossEntropyLoss()
        clf_loss = class_loss(class_output, y)
        if run is not None:
            run["metrics/" + mode + "/clf_loss"].log(clf_loss)
        return clf_loss

    def train_round(self, dataloader, kl_coeff, clf_coeff, optimizer, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        lr
            learning rate
        kl_coeff 
            coefficient for weighting Kullback-Leibler loss
        clf_coeff
            coefficient for weighting classifier loss
        optimizer
            optimizer for training
        run
            Neptune run if training is to be logged
        """
        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move batch to device
            data = minibatch[0].to(self.device)
            batch = minibatch[2].to(self.device)
            if self.batches is None:
                batch=None

            # reset optimizer
            optimizer.zero_grad()

            # forward step
            if self.labels is not None:
                reconstruction, mu, log_var, output = self.forward(data, batch)
            else:
                reconstruction, mu, log_var = self.forward(data, batch)

            # calculate VAE loss
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            
            # add classifier loss
            if self.labels is not None:
                class_loss = self.classify_loss(output, minibatch[1].to(self.device), mode='train', run=run)
                loss += clf_coeff * class_loss

            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            if self.decoder.decoder[0][0].weight.requires_grad:
                for i in range(len(self.decoder.decoder)):
                    self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            if self.decoder.decoder[0][0].weight.requires_grad:
                for i in range(len(self.decoder.decoder)):
                    self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    def val_round(self, dataloader, kl_coeff, clf_coeff, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        clf_coeff
            coefficient for weighting classifier loss
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        with torch.no_grad():
            # iterate over dataloader for validation
            for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

                # move batch to device
                data = minibatch[0].to(self.device)
                batch = minibatch[2].to(self.device)
                if self.batches is None:
                    batch=None

                # forward step
                if self.labels is not None:
                    reconstruction, mu, log_var, output = self.forward(data, batch)
                else:
                    reconstruction, mu, log_var = self.forward(data, batch)

                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, mode='val', run=run)
                
                # add classification loss
                if self.labels is not None:
                    class_loss = self.classify_loss(output, minibatch[1].to(self.device), mode='val', run=run)
                    loss += clf_coeff * class_loss

                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, modelpath, lr=1e-4, kl_coeff=1e-4, clf_coeff=1e5, batch_size=128, epochs=300, run=None):
        """
        Parameters
        ----------
        modelpath
            where to store the best model (full path with filename)
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        clf_coeff
            coefficient for weighting classifier loss
        batch_size
            size of minibatches
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """
        # train-test split
        indices = np.random.RandomState(seed=42).permutation(self.X.shape[0])
        train_ind = indices[:round(len(indices)*0.8)]
        val_ind = indices[round(len(indices)*0.8):]
        X_train, y_train, batch_train = self.X[train_ind,:], self.y[train_ind], self.batch_info[train_ind]
        X_val, y_val, batch_val = self.X[val_ind,:], self.y[val_ind], self.batch_info[val_ind]

        # convert train and val into torch tensors
        X_train, y_train, batch_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train), torch.tensor(batch_train)
        X_val, y_val, batch_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val), torch.tensor(batch_val)

        # generate dataloaders
        trainloader = FastTensorDataLoader(X_train, 
                                           y_train,
                                           batch_train,
                                       batch_size=batch_size, 
                                       shuffle=True)
        valloader = FastTensorDataLoader(X_val, 
                                         y_val,
                                         batch_val,
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, clf_coeff, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, clf_coeff, run)
            
            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_epoch_loss,
                    'classifier_trained': True if self.labels is not None else False,
                    'n_classes': self.n_classes,
                    'batch_corrected': True if self.n_batch > 0 else False,
                    'n_batch': self.n_batch
                }, modelpath)
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")


    def _get_activation(self, index, activation={}):
        def hook(model, input, output):
            activation[index] = output
        return hook

    def _attach_hooks(self, activation={}, hooks={}):
        for i in range(len(self.decoder.decoder)-1):
            key = str(i)
            value = self.decoder.decoder[i][0].register_forward_hook(self._get_activation(i, activation))
            hooks[key] = value


    def _pass_data(self, data, batch=None, output='act'):
        """
        data
            the dataset to pass through model
        batch
            batch information of the data
        output
            one of 'act': pathway activities
                    'rec': reconstructed values
        """

        # set to eval mode
        self.eval()

        # get latent space embedding
        with torch.no_grad():
            z = self.get_embedding(data, batch)
            z = z.to('cpu').detach().numpy()
        
        z = np.array(np.split(z, z.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T

        # initialize activations and hooks
        activation = {}
        hooks = {}

        # attach the hooks
        self._attach_hooks(activation=activation, hooks=hooks)
        
        # pass data through model
        with torch.no_grad():
            if self.labels is not None:
                reconstruction, _, _, _ = self.forward(data, batch)
            else:
                reconstruction, _, _ = self.forward(data, batch)

        act = torch.cat(list(activation.values()), dim=1).to('cpu').detach().numpy()
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if output == 'act':
            return np.hstack((z,act))
        if output == 'rec':
            return reconstruction.to('cpu').detach().numpy()
        

    def get_pathway_activities(self, ontobj, dataset, batch=None, terms=None):
        """
        Retrieves pathway activities from latent space and decoder.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for pathway activity retrieval
        batch
            batch information of the dataset
        terms
            list of ontology term ids whose activities should be retrieved
        """
        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retrieve pathway activities
        act = self._pass_data(data, batch=batch, output='act')

        # if term was specified, subset
        if terms is not None:
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            act = act[:,term_ind]

        return act


    def get_reconstructed_values(self, ontobj, dataset, batch=None, rec_genes=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for pathway activity retrieval
        batch
            batch information of the dataset
        rec_genes
            list of genes whose reconstructed values should be retrieved
        """
        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retrieve pathway activities
        rec = self._pass_data(data, batch=batch, output='rec')

        # if genes were passed, subset
        if rec_genes is not None:
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            rec = rec[:,gene_ind]

        return rec

        
    def perturbation(self, ontobj, dataset, genes, values, batch=None, output='terms', terms=None, rec_genes=None):
        """
        Retrieves pathway activities or reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for perturbation and pathway activity retrieval
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        batch
            batch information of the dataset
        output
            - 'terms': retrieve pathway activities
            - 'genes': retrieve reconstructed values

        terms
            list of ontology term ids whose values should be retrieved
        rec_genes
            list of genes whose values should be retrieved
        """

        if output == 'terms':
            rec_genes=None
        if output == ' genes':
            terms = None

        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # get indices of the genes in list
        indices = [self.genes.index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            data[:,indices[i]] = values[i]

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # get pathway activities or reconstructed values after perturbation
        if output == 'terms':
            res = self._pass_data(data, batch=batch, output='act')
        if output == 'genes':
            res = self._pass_data(data, batch=batch, output='rec')

        # if term was specified, subset
        if terms is not None:
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            res = res[:,term_ind]
        
        if rec_genes is not None:
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            res = res[:,gene_ind]

        return res
        





###-------------------------------------------------------------###
##               VAE WITH ONTOLOGY IN ENCODER                    ##
###-------------------------------------------------------------###


class OntoEncVAE(nn.Module):
    """
    This class combines a an ontology structured encoder with a normal decoder

    Parameters
    ----------
    ontobj
        instance of the class Ontobj(), containing a preprocessed ontology and the training data
    dataset
        which dataset from Ontobj to use for training
    top_thresh
        top threshold to tell which trimmed ontology to use
    bottom_thresh
        bottom_threshold to tell which trimmed ontology to use
    neuronnum
        number of neurons per term
    drop
        dropout rate, default is 0.2
    z_drop
        dropout rate for latent space, default is 0.5
    """

    def __init__(self, ontobj, dataset, top_thresh=1000, bottom_thresh=30, neuronnum=3, drop=0.2, z_drop=0.5):
        super(OntoEncVAE, self).__init__()

        self.ontology = ontobj.description
        self.top = top_thresh
        self.bottom = bottom_thresh
        self.genes = ontobj.genes[str(top_thresh) + '_' + str(bottom_thresh)]
        self.in_features = len(self.genes)
        self.mask_list = ontobj.masks[str(top_thresh) + '_' + str(bottom_thresh)]['encoder']
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_enc = np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.latent_dim = self.layer_dims_enc[-1] * neuronnum
        self.layer_dims_dec = [self.latent_dim]
        self.neuronnum = neuronnum
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder
        self.encoder = OntoEncoder(self.in_features,
                                    self.layer_dims_enc,
                                    self.mask_list,
                                    self.latent_dim,
                                    self.neuronnum)

        # Decoder
        self.decoder = Decoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_dec,
                                self.drop,
                                self.z_drop)

        if not dataset in ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys():
            raise ValueError('Available datasets are: ' + ', '.join(list(ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys())))

        self.X = ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)][dataset]

    def reparameterize(self, mu, log_var):
        """
        Parameters
        ----------
        mu
            mean from the encoder's latent space
        log_var
            log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            dataset of which embedding should be generated
        """
        # set to eval mode
        self.eval()

        # run data through encoder
        mu, log_var = self.encoder(x)
        embedding = self.reparameterize(mu, log_var)
        return embedding

    def forward(self, x):
        # encoding
        mu, log_var = self.encoder(x)
            
        # sample from latent space
        z = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(z)
            
        return reconstruction, mu, log_var

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, mode='train', run=None):
        """
        Calculates VAE loss as combination of reconstruction loss and weighted Kullback-Leibler loss.
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def train_round(self, dataloader, lr, kl_coeff, optimizer, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        lr
            learning rate
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        optimizer
            optimizer for training
        """
        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move batch to device
            data = data[0].to(self.device)
            optimizer.zero_grad()

            # forward step
            reconstruction, mu, log_var = self.forward(data)
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(len(self.encoder.encoder)):
                self.encoder.encoder[i][0].weight.grad = torch.mul(self.encoder.encoder[i][0].weight.grad, self.encoder.masks[i])

            # apply mask on latent space
            self.encoder.mu[0].weight.grad = torch.mul(self.encoder.mu[0].weight.grad, self.encoder.masks[-1])
            self.encoder.logvar[0].weight.grad = torch.mul(self.encoder.logvar[0].weight.grad, self.encoder.masks[-1])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            for i in range(len(self.encoder.encoder)):
                self.encoder.encoder[i][0].weight.data = self.encoder.encoder[i][0].weight.data.clamp(0)
            self.encoder.mu[0].weight.data = self.encoder.mu[0].weight.data.clamp(0)
            self.encoder.logvar[0].weight.data = self.encoder.logvar[0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    def val_round(self, dataloader, kl_coeff, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        with torch.no_grad():
            # iterate over dataloader for validation
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

                # move batch to device
                data = data[0].to(self.device)

                # forward step
                reconstruction, mu, log_var = self.forward(data)
                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, mode='val', run=run)
                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, modelpath, lr=1e-4, kl_coeff=1e-4, batch_size=128, epochs=300, run=None):
        """
        Parameters
        ----------
        modelpath
            where to store the best model (full path with filename)
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        batch_size
            size of minibatches
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """
        # train-test split
        indices = np.random.RandomState(seed=42).permutation(self.X.shape[0])
        X_train_ind = indices[:round(len(indices)*0.8)]
        X_val_ind = indices[round(len(indices)*0.8):]
        X_train, X_val = self.X[X_train_ind,:], self.X[X_val_ind,:]

        # convert train and val into torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        # generate dataloaders
        trainloader = FastTensorDataLoader(X_train, 
                                       batch_size=batch_size, 
                                       shuffle=True)
        valloader = FastTensorDataLoader(X_val, 
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, lr, kl_coeff, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, run)
            
            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_epoch_loss,
                }, modelpath)
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")

    def _pass_data(self, data, output):
        """
        Passes data through the model.

        Parameters
        ----------
        data
            data to be passed
        output
            'act': return pathway activities
            'rec': return reconstructed values
        """

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # set to eval mode
        self.eval()

        # get latent space embedding
        with torch.no_grad():
            z = self.get_embedding(data)
            z = z.to('cpu').detach().numpy()
        
        z = np.array(np.split(z, z.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T

        # get activities from encoder
        activation = {}
        def get_activation(index):
            def hook(model, input, output):
                activation[index] = output.to('cpu').detach()
            return hook

        hooks = {}

        for i in range(len(self.encoder.encoder)):
            key = str(i)
            value = self.encoder.encoder[i][0].register_forward_hook(get_activation(i))
            hooks[key] = value
        
        with torch.no_grad():
            reconstruction, _, _ = self.forward(data)

        act = torch.cat(list(activation.values())[::-1], dim=1).detach().numpy()
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if output == 'act':
            return np.hstack((z,act))
        if output == 'rec':
            return reconstruction.to('cpu').detach().numpy()


    def get_pathway_activities(self, ontobj, dataset, terms=None):
        """
        Retrieves pathway activities from encoder and latent space.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for pathway activity retrieval
        terms
            list of ontology term ids whose activities should be retrieved
        """
        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retrieve pathway activities
        act = self._pass_data(data, 'act')

        # if term was specified, subset
        if terms is not None:
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            act = act[:,term_ind]

        return act

        
    def get_reconstructed_values(self, ontobj, dataset, rec_genes=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for pathway activity retrieval
        rec_genes
            list of genes whose reconstructed values should be retrieved
        """
        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retrieve pathway activities
        rec = self._pass_data(data, 'rec')

        # if genes were passed, subset
        if rec_genes is not None:
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            rec = rec[:,gene_ind]

        return rec


    def perturbation(self, ontobj, dataset, genes, values, output='terms', terms=None, rec_genes=None):
        """
        Retrieves pathway activities or reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        ontobj
            instance of the class Ontobj(), should be the same as the one used for model training
        dataset
            which dataset to use for perturbation and pathway activity retrieval
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        output
            - 'terms': retrieve pathway activities
            - 'genes': retrieve reconstructed values

        terms
            list of ontology term ids whose values should be retrieved
        rec_genes
            list of genes whose values should be retrieved
        """

        if self.ontology != ontobj.description:
            raise ValueError('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # get indices of the genes in list
        indices = [self.genes.index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            data[:,indices[i]] = values[i]

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # get pathway activities or reconstructed values after perturbation
        if output == 'terms':
            res = self._pass_data(data, 'act')
        if output == 'genes':
            res = self._pass_data(data, 'rec')

        # if term was specified, subset
        if terms is not None:
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            res = res[:,term_ind]
        
        if rec_genes is not None:
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            res = res[:,gene_ind]

        return res



###-------------------------------------------------------------###
##                              VAE                              ##
###-------------------------------------------------------------###

class VAE(nn.Module):
    """
    This class defines a standard VAE without ontology.

    Parameters
    ----------
    in_features
        data to be used for training, 2D numpy array with samples in rows and features in columns
    layer_dims_enc
        list giving the dimensions of the layers in the encoder
    layer_dims_dec
        list giving the dimensions of the layers in the decoder
    drop
        dropout rate, default is 0
    z_drop
        dropout rate for latent space, default is 0.5
    labels
        labels of samples if classifier is used
    batches
        batch information of samples if available
    """

    def __init__(
        self, 
        data, 
        layer_dims_enc=[1000], 
        layer_dims_dec=[1000], 
        latent_dim=128, 
        drop=0.2, 
        z_drop=0.5,
        labels=None,
        batches=None,
        modelpath=None
        ):

        super(VAE, self).__init__()

        self.X = data
        self.in_features = self.X.shape[1]
        self.layer_dims_enc = layer_dims_enc
        self.layer_dims_dec = layer_dims_dec
        self.latent_dim = latent_dim
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = labels
        self.batches = batches
        self.one_hot = None
        self.VAE_trained = False
        self.clf_trained = False
        self.modelpath = modelpath

        # set labels (or dummies) for classification
        if self.labels is not None:
            self.y = self.labels
        else:
            self.y = np.zeros(self.X.shape[0])
        self.n_classes = len(set(self.y))

        # set batches (or dummies) for batch correction
        if self.batches is not None:
            self.n_batch = len(set(self.batches))
            self.one_hot = torch.eye(self.n_batch).to(self.device)
            self.batch_info = self.batches
        else:
            self.n_batch = 0
            self.batch_info = np.zeros(self.X.shape[0])

        # parse model information if pretrained model was passed
        if self.modelpath is not None:
            checkpoint = torch.load(modelpath,
                        map_location = torch.device(self.device))
            self.VAE_trained = True
            self.val_loss_min = checkpoint['loss']
            if checkpoint['classifier_trained']:
                self.clf_trained = True
                self.n_classes = checkpoint['n_classes']
            if checkpoint['batch_corrected']:
                self.n_batch = checkpoint['n_batch']
                self.one_hot = torch.eye(self.n_batch).to(self.device)
        # Encoder
        self.encoder = Encoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_enc,
                                self.drop,
                                self.z_drop)
        
        # Decoder
        self.decoder = Decoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_dec,
                                self.drop)

        # Classifier (optional)
        if self.labels is not None:
            self.classifier = simple_classifier(in_features=self.latent_dim,
                                             n_classes=self.n_classes)

        # load parameters if pretrained model was passed
        if self.modelpath is not None:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)   
        
    def reparameterize(self, mu, log_var):
        """
        Parameters
        ----------
        mu
            mean from the encoder's latent space
        log_var
            log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x, batch=None):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            dataset of which embedding should be generated
        """
        # attach batch information
        if batch is not None:
            x = torch.hstack((x, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            x = torch.hstack((x, torch.zeros((x.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # set to eval mode
        self.eval()

        # encoding
        mu, log_var = self.encoder(x)

        # sample from latent space
        embedding = self.reparameterize(mu, log_var)
        return embedding

    def forward(self, x, batch=None):

        # attach batch information
        if batch is not None:
            x = torch.hstack((x, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            x = torch.hstack((x, torch.zeros((x.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # encoding
        mu, log_var = self.encoder(x)
            
        # sample from latent space
        z = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(z)
        
        if self.labels is not None:
            output = self.classifier(z)
            return reconstruction, mu, log_var, output
        else:
            return reconstruction, mu, log_var
    
    def get_classification(self, dataset, batch=None):
        """
        Parameters
        ----------
        dataset
            which dataset to use for pathway activity retrieval
        """
        # convert data to tensor 
        data = torch.tensor(dataset, dtype=torch.float32).to(self.device)

        # attach batch information
        if batch is not None:
            data = torch.hstack((data, self.one_hot[batch]))
        if batch is None and self.n_batch > 0:
            data = torch.hstack((data, torch.zeros((data.shape[0])).repeat(self.n_batch,1).T.to(self.device)))

        # set to eval mode
        self.eval()

        with torch.no_grad():
            # encoding
            mu, log_var = self.encoder(data)

            # sample from latent space
            z = self.reparameterize(mu, log_var)

        # perform classification
        with torch.no_grad():
            output = self.classifier(z)

        output = output.to('cpu').detach().numpy()
        y_pred = np.argmax(output,axis=1)

        return y_pred

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, mode='train', run=None):
        """
        Parameters
        ----------
        mode
            'train' or 'val' (default): for logging
        run
            Neptune run if training is to be logged
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def classify_loss(self, class_output, y, mode='val', run=None):
        """
        Parameters
        ----------
        mode
            'train' or 'val' (default): for logging
        run
            Neptune run if training is to be logged
        """
        class_loss = nn.CrossEntropyLoss()
        clf_loss = class_loss(class_output, y)
        if run is not None:
            run["metrics/" + mode + "/clf_loss"].log(clf_loss)
        return clf_loss

    def train_round(self, dataloader, lr, kl_coeff, optimizer, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        lr
            learning rate
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        optimizer
            optimizer for training
        run
            Neptune run if training is to be logged
        """
        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move batch to device
            data = minibatch[0].to(self.device)
            batch = minibatch[2].to(self.device)
            if self.batches is None:
                batch=None

            # reset optimizer
            optimizer.zero_grad()

            # forward step
            if self.labels is not None:
                reconstruction, mu, log_var, output = self.forward(data, batch)
            else:
                reconstruction, mu, log_var = self.forward(data, batch)
            
            # calculate VAE loss
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)

            # add classifier loss
            if self.labels is not None:
                class_loss = self.classify_loss(output, minibatch[1].to(self.device), mode='train', run=run)
                loss += clf_coeff * class_loss

            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # perform optimizer step
            optimizer.step()

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    def val_round(self, dataloader, kl_coeff, run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        with torch.no_grad():
            # iterate over dataloader for validation
            for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

                # move batch to device
                data = minibatch[0].to(self.device)
                batch = minibatch[2].to(self.device)
                if self.batches is None:
                    batch=None

                # forward step
                if self.labels is not None:
                    reconstruction, mu, log_var, output = self.forward(data, batch)
                else:
                    reconstruction, mu, log_var = self.forward(data, batch)

                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, mode='val', run=run)
                
                # add classification loss
                if self.labels is not None:
                    class_loss = self.classify_loss(output, minibatch[1].to(self.device), mode='val', run=run)
                    loss += clf_coeff * class_loss

                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, modelpath, lr=1e-4, kl_coeff=1e-4, clf_coeff=1e5, batch_size=128, epochs=300, run=None):
        """
        Parameters
        ----------
        modelpath
            where to store the best model (full path with filename)
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        clf_coeff
            coefficient for weighting classifier loss
        batch_size
            size of minibatches
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """
        # train-test split
        indices = np.random.RandomState(seed=42).permutation(self.X.shape[0])
        train_ind = indices[:round(len(indices)*0.8)]
        val_ind = indices[round(len(indices)*0.8):]
        X_train, y_train, batch_train = self.X[train_ind,:], self.y[train_ind], self.batch_info[train_ind]
        X_val, y_val, batch_val = self.X[val_ind,:], self.y[val_ind], self.batch_info[val_ind]

        # convert train and val into torch tensors
        X_train, y_train, batch_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train), torch.tensor(batch_train)
        X_val, y_val, batch_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val), torch.tensor(batch_val)

        # generate dataloaders
        trainloader = FastTensorDataLoader(X_train, 
                                           y_train,
                                           batch_train,
                                       batch_size=batch_size, 
                                       shuffle=True)
        valloader = FastTensorDataLoader(X_val, 
                                         y_val,
                                         batch_val,
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, lr, kl_coeff, clf_coeff, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, clf_coeff, run)
            
            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_epoch_loss,
                    'classifier_trained': True if self.labels is not None else False,
                    'n_classes': self.n_classes,
                    'batch_corrected': True if self.n_batch > 0 else False,
                    'n_batch': self.n_batch
                }, modelpath)
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")



