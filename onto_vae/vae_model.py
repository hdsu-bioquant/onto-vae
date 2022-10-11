#!/usr/bin/env python3

import sys
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .modules import Encoder, Decoder, OntoEncoder, OntoDecoder
from .fast_data_loader import FastTensorDataLoader

###-------------------------------------------------------------###
##                  VAE WITH ONTOLOGY IN DECODER                 ##
###-------------------------------------------------------------###

class OntoVAE(nn.Module):
    """
    This class combines a normal encoder with an ontology structured decoder

    Parameters
    -------------
    ontobj: instance of the class Ontobj(), containing a preprocessed ontology and the training data
    dataset: which dataset to use for training
    top_thresh: top threshold to tell which trimmed ontology to use
    bottom_thresh: bottom_threshold to tell which trimmed ontology to use
    neuronnum: number of neurons per term
    drop: dropout rate, default is 0.2
    z_drop: dropout rate for latent space, default is 0.5
    """

    def __init__(self, ontobj, dataset, top_thresh=1000, bottom_thresh=30, neuronnum=3, drop=0.2, z_drop=0.5):
        super(OntoVAE, self).__init__()

        if not str(top_thresh) + '_' + str(bottom_thresh) in ontobj.genes.keys():
            sys.exit('Available trimming thresholds are: ' + ', '.join(list(ontobj.genes.keys())))

        self.ontology = ontobj.description
        self.top = top_thresh
        self.bottom = bottom_thresh
        self.genes = ontobj.genes[str(top_thresh) + '_' + str(bottom_thresh)]
        self.in_features = len(self.genes)
        self.mask_list = ontobj.masks[str(top_thresh) + '_' + str(bottom_thresh)]
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.latent_dim = self.layer_dims_dec[0] * neuronnum
        self.layer_dims_enc = [self.latent_dim]
        self.neuronnum = neuronnum
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder
        self.encoder = Encoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_enc,
                                self.drop,
                                self.z_drop)

        # Decoder
        self.decoder = OntoDecoder(self.in_features,
                                    self.layer_dims_dec,
                                    self.mask_list,
                                    self.latent_dim,
                                    self.neuronnum,
                                    self.drop)
        
        if not dataset in ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys():
            sys.exit('Available datasets are: ' + ', '.join(list(ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)].keys())))

        self.X = ontobj.data[str(top_thresh) + '_' + str(bottom_thresh)][dataset]

    def reparameterize(self, mu, log_var):
        """
        Parameters
        -------------
        mu: mean from the encoder's latent space
        log_var: log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x):
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

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff):
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def train_round(self, dataloader, lr, kl_coeff, optimizer):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        lr: learning rate
        kl_coeff: coefficient for weighting Kullback-Leibler loss
        optimizer: optimizer for training
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
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            for i in range(len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    def val_round(self, dataloader, kl_coeff):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        kl_coeff: coefficient for weighting Kullback-Leibler loss
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
                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff)
                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, modelpath, lr=1e-4, kl_coeff=1e-4, batch_size=128, epochs=300, log=True, **kwargs):
        """
        Parameters
        -------------
        modelpath: where to store the best model (full path with filename)
        lr: learning rate
        kl_coeff: Kullback Leibler loss coefficient
        batch_size: size of minibatches
        epochs: over how many epochs to train
        log: whether run should be logged to neptune
        **kwargs: pass the run here if log == True
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
            train_epoch_loss = self.train_round(trainloader, lr, kl_coeff, optimizer)
            val_epoch_loss = self.val_round(valloader, kl_coeff)
            
            if log == True:
                run = kwargs.get('run')
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
        output
            one of 'act': pathway activities
                    'rec': reconstructed values
        """

        # set to eval mode
        self.eval()

        # get latent space embedding
        with torch.no_grad():
            z = self.get_embedding(data)
            z = z.to('cpu').detach().numpy()
        
        z = np.array(np.split(z, z.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T

        # get activities from decoder
        activation = {}
        def get_activation(index):
            def hook(model, input, output):
                activation[index] = output.to('cpu').detach()
            return hook

        hooks = {}

        for i in range(len(self.decoder.decoder)-1):
            key = str(i)
            value = self.decoder.decoder[i][0].register_forward_hook(get_activation(i))
            hooks[key] = value
        
        with torch.no_grad():
            reconstruction, _, _ = self.forward(data)

        act = torch.cat(list(activation.values()), dim=1).numpy()
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if output == 'act':
            return np.hstack((z,act))
        if output == 'rec':
            return reconstruction.to('cpu').detach().numpy()
        

    def get_pathway_activities(self, ontobj, dataset, **kwargs):
        """
        Parameters
        -------------
        ontobj: instance of the class Ontobj(), should be the same as the one used for model training
        dataset: which dataset to use for pathway activity retrieval
        **kwargs
        terms: if we only want to get back the activities for certain terms (should be list of ids)
        """
        if self.ontology != ontobj.description:
            sys.exit('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retireve pathway activities
        act = self._pass_data(data, 'act')

        # if term was specified, subset
        if 'terms' in kwargs:
            terms = kwargs.get('terms')
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            act = act[:,term_ind]

        return act


    def get_reconstructed_values(self, ontobj, dataset, **kwargs):
        """
        Parameters
        -------------
        ontobj: instance of the class Ontobj(), should be the same as the one used for model training
        dataset: which dataset to use for pathway activity retrieval

        **kwargs
        rec_genes: if we only want to get back values for certain genes
        """
        if self.ontology != ontobj.description:
            sys.exit('Wrong ontology provided, should be ' + self.ontology)

        data = ontobj.data[str(self.top) + '_' + str(self.bottom)][dataset].copy()

        # convert data to tensor and move to device
        data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # retrieve pathway activities
        rec = self._pass_data(data, 'rec')

        # if genes were passed, subset
        if 'rec_genes' in kwargs:
            genes = kwargs.get('rec_genes')
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in genes])

            rec = rec[:,gene_ind]

        return rec

        
    def perturbation(self, ontobj, dataset, genes, values, output='terms', **kwargs):
        """
        This function retrieves pathway activities after performing in silico perturbation

        Parameters
        -------------
        ontobj: instance of the class Ontobj(), should be the same as the one used for model training
        dataset: which dataset to use for perturbation and pathway activity retrieval
        genes: a list of genes to perturb
        values: list with new values, same length as genes
        output: 'terms' or 'genes'

        **kwargs
        terms: if we only want to get back the activities for certain terms (should be list of ids)
        rec_genes: if we only want to get back values for certain reconstructed genes
        """

        if self.ontology != ontobj.description:
            sys.exit('Wrong ontology provided, should be ' + self.ontology)

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
        if 'terms' in kwargs:
            terms = kwargs.get('terms')
            annot = ontobj.annot[str(self.top) + '_' + str(self.bottom)]
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            res = res[:,term_ind]
        
        if 'rec_genes' in kwargs:
            genes = kwargs.get('rec_genes')
            onto_genes = ontobj.genes[str(self.top) + '_' + str(self.bottom)]
            gene_ind = np.array([onto_genes.index(g) for g in genes])

            res = res[:,gene_ind]

        return res
        





###-------------------------------------------------------------###
##               VAE WITH ONTOLOGY IN ENCODER                    ##
###-------------------------------------------------------------###

# This class has not been updated yet to work with an Ontoobj, so use carefully.

class OntoEncVAE(nn.Module):
    """
    This class combines a an ontology structured encoder with a normal decoder

    Parameters
    -------------
    in_features: # of features that are used as input
    mask_list: matrix for each layer transition, that determines which weights to zero out
    neuronnum: number of neurons per term
    drop: dropout rate, default is 0
    z_drop: dropout rate for latent space, default is 0.5
    """

    def __init__(self, onto, in_features, mask_list, neuronnum=1, drop=0, z_drop=0.5):
        super(OntoEncVAE, self).__init__()

        self.in_features = in_features
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in mask_list]
        self.layer_dims_enc = np.array([mask_list[0].shape[1]] + [m.shape[0] for m in mask_list])
        self.latent_dim = layer_dims_enc[-1] * neuronnum
        self.layer_dims_dec = [self.latent_dim]
        self.neuronnum = neuronnum
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder
        self.encoder = OntoEncoder(self.in_features,
                                    self.layer_dims_enc,
                                    self.mask_list,
                                    self.drop,
                                    self.z_drop)

        # Decoder
        self.decoder = Decoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_dec,
                                self.drop)
        
    def reparameterize(self, mu, log_var):
        """
        Parameters
        -------------
        mu: mean from the encoder's latent space
        log_var: log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x):
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

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff):
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def train_round(self, dataloader, lr, kl_coeff, optimizer):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        lr: learning rate
        kl_coeff: coefficient for weighting Kullback-Leibler loss
        optimizer: optimizer for training
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
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff)
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

    def val_round(self, dataloader, kl_coeff):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        kl_coeff: coefficient for weighting Kullback-Leibler loss
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
                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff)
                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, trainloader, valloader, lr, kl_coeff, epochs, modelpath, log=True, **kwargs):
        """
        Parameters
        -------------
        trainloader: pytorch dataloader instance with training data
        valloader: pytorch dataloader instance with validation data
        lr: learning rate
        kl_coeff: coefficient for weighting Kullback-Leibler loss
        epochs: number of epochs to train the model
        modelpath: where to store the best model
        log: if losses should be logged
        **kwargs: pass the run here if log == True
        """
        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, lr, kl_coeff, optimizer)
            val_epoch_loss = self.val_round(valloader, kl_coeff)
            
            if log == True:
                run = kwargs.get('run')
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


    def get_pathway_activities(self, data):
        """
        Parameters
        -------------
        2D numpy array to be run through trained model
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

        # get activities from decoder
        activation = {}
        def get_activation(index):
            def hook(model, input, output):
                activation[index] = output.to('cpu').detach()
            return hook

        hooks = {}

        for i in range(len(self.encoder.encoder)-1):
            key = 'Dec' + str(i)
            value = self.encoder.encoder[i][0].register_forward_hook(get_activation(i))
            hooks[key] = value
        
        with torch.no_grad():
            reconstruction, _, _ = self.forward(data)

        act = torch.cat(list(activation.values()), dim=1).detach().numpy()
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T

        # stack and return
        return np.hstack((z,act))




###-------------------------------------------------------------###
##                              VAE                              ##
###-------------------------------------------------------------###

class VAE(nn.Module):
    """
    This class defines a standard VAE without ontology.

    Parameters
    -------------
    in_features: # of features that are used as input
    layer_dims_enc: list giving the dimensions of the layers in the encoder
    layer_dims_dec: list giving the dimensions of the layers in the decoder
    drop: dropout rate, default is 0
    z_drop: dropout rate for latent space, default is 0.5
    """

    def __init__(self, in_features, layer_dims_enc=[1000], layer_dims_dec=[1000], latent_dim=500, drop=0, z_drop=0.5):
        super(VAE, self).__init__()

        self.in_features = in_features
        self.layer_dims_enc = layer_dims_enc
        self.layer_dims_dec = layer_dims_dec
        self.latent_dim = latent_dim
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        
    def reparameterize(self, mu, log_var):
        """
        Parameters
        -------------
        mu: mean from the encoder's latent space
        log_var: log variance from the encoder's latent space
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
    def get_embedding(self, x):
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

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff):
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def train_round(self, dataloader, lr, kl_coeff, optimizer):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        lr: learning rate
        kl_coeff: coefficient for weighting Kullback-Leibler loss
        optimizer: optimizer for training
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
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # perform optimizer step
            optimizer.step()

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    def val_round(self, dataloader, kl_coeff):
        """
        Parameters
        -------------
        dataloader: pytorch dataloader instance with training data
        kl_coeff: coefficient for weighting Kullback-Leibler loss
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
                loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff)
                running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, trainloader, valloader, lr, kl_coeff, epochs, modelpath, log=True, **kwargs):
        """
        Parameters
        -------------
        trainloader: pytorch dataloader instance with training data
        valloader: pytorch dataloader instance with validation data
        lr: learning rate
        kl_coeff: coefficient for weighting Kullback-Leibler loss
        epochs: number of epochs to train the model
        modelpath: where to store the best model
        log: if losses should be logged
        **kwargs: pass the run here if log == True
        """
        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, lr, kl_coeff, optimizer)
            val_epoch_loss = self.val_round(valloader, kl_coeff)
            
            if log == True:
                run = kwargs.get('run')
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



