#!/usr/bin/env python3

# Modules used by OntoVAE

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F



###-------------------------------------------------------------###
##                        CLASSIFIER                             ##
###-------------------------------------------------------------###

class simple_classifier(nn.Module):
    """
    This classifier takes in the pathway activities and performs classification

    Parameters
    -------------
    in_features: dimension of decoder pathway activities
    n_classes: number of classes
    """

    def __init__(self, in_features, n_classes, drop=0.5):
        super(simple_classifier, self).__init__()

        self.in_features = in_features
        self.n_classes = n_classes
        self.drop=drop
     
        self.classifier = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features, self.n_classes),
                    nn.Dropout(p=self.drop),
                    nn.Sigmoid()
                )
            ] 
        )

    def forward(self, x):

        c = x
        for layer in self.classifier:
            c = layer(c)

        return c


###-------------------------------------------------------------###
##                       ENCODER CLASS                           ##
###-------------------------------------------------------------###

class Encoder(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.

    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list giving the dimensions of the hidden layers
    latent_dim 
        latent dimension
    drop
        dropout rate, default is 0
    z_drop
        dropout rate for latent space, default is 0.5
    """

    def __init__(self, in_features, latent_dim, layer_dims=[512], drop=0, z_drop=0.5):
        super(Encoder, self).__init__()

        self.in_features = in_features
        self.layer_dims = layer_dims
        self.layer_nums = [layer_dims[i:i+2] for i in range(len(layer_dims)-1)]
        self.latent_dim = latent_dim
        self.drop = drop
        self.z_drop = z_drop

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features, self.layer_dims[0]),
                    nn.BatchNorm1d(self.layer_dims[0]),
                    nn.Dropout(p=self.drop),
                    nn.ReLU()
                )
            ] +

            [self.build_block(x[0], x[1]) for x in self.layer_nums] 
        )

        self.mu = nn.Sequential(
            nn.Linear(self.layer_dims[-1], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        )

        self.logvar = nn.Sequential(
            nn.Linear(self.layer_dims[-1], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        )

    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=self.drop),
            nn.ReLU()
        )

    def forward(self, x):

        # encoding
        c = x
        for layer in self.encoder:
            c = layer(c)

        mu = self.mu(c)
        log_var = self.logvar(c)

        return mu, log_var



###-------------------------------------------------------------###
##                       DECODER CLASS                           ##
###-------------------------------------------------------------###



class Decoder(nn.Module):
    """
    This class constructs a Decoder module for a variational autoencoder.

    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list giving the dimensions of the hidden layers
    latent_dim
        latent dimension, default is 128
    drop
        dropout rate, default is 0
    """

    def __init__(self, in_features, latent_dim, layer_dims=[512], drop=0, z_drop=0.5):
        super(Decoder, self).__init__()

        self.in_features = in_features
        self.layer_dims = layer_dims
        self.layer_nums = [layer_dims[i:i+2] for i in range(len(layer_dims)-1)]
        self.latent_dim = latent_dim
        self.drop = drop

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.layer_dims[0]),
                    nn.BatchNorm1d(self.layer_dims[0]),
                    nn.Dropout(p=self.drop),
                    nn.ReLU()
                )
            ] +

            [self.build_block(x[0], x[1]) for x in self.layer_nums] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_dims[-1], self.in_features),
                )
            ]
        )

    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=self.drop),
            nn.ReLU()
        )

    def forward(self, z):

        # decoding
        reconstruction = z

        for layer in self.decoder:
            reconstruction = layer(reconstruction)
        
        return reconstruction


###-------------------------------------------------------------###
##                     ONTO ENCODER CLASS                        ##
###-------------------------------------------------------------###

class OntoEncoder(nn.Module):
    """
    This class constructs an ontology structured Encoder module.

    Parameters
    ----------
    in_features
        of features that are used as input
    layer_dims
        list of tuples that define in and out for each layer
    mask_list
        matrix for each layer transition, that determines which weights to zero out
    drop
        dropout rate, default is 0
    z_drop
        dropout rate for latent space, default is 0.5
    """ 

    def __init__(self, in_features, layer_dims, mask_list, latent_dim, neuronnum=3):
        super(OntoEncoder, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.layer_dims = layer_dims
        self.layer_shapes = [(np.sum(self.layer_dims[:i+1]), self.layer_dims[i+1]) for i in range(len(self.layer_dims)-1)]
        self.masks = []
        for m in mask_list:
            self.masks.append(m.to(self.device))
        self.latent_dim = self.layer_dims[-1]
        self.drop = drop
        self.z_drop = z_drop

        # Encoder
        self.encoder = nn.ModuleList(

            [self.build_block(x[0], x[1]) for x in self.layer_shapes[:-1]] 

        ).to(self.device)

        self.mu = nn.Sequential(
            nn.Linear(self.layer_shapes[-1][0], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        ).to(self.device)

        self.logvar = nn.Sequential(
            nn.Linear(self.layer_shapes[-1][0], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        ).to(self.device)

        # apply masks to zero out weights of non-existing connections
        for i in range(len(self.encoder)):
            self.encoder[i][0].weight.data = torch.mul(self.encoder[i][0].weight.data, self.masks[i])

        # apply mask on latent space
        self.mu[0].weight.data = torch.mul(self.mu[0].weight.data, self.masks[-1])
        self.logvar[0].weight.data = torch.mul(self.logvar[0].weight.data, self.masks[-1])


    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs),
            nn.Dropout(p=self.drop)
        )
    

    def forward(self, x):
        
        # encoding
        out = x

        for layer in self.encoder:
            c = layer(out)
            out = torch.cat((c, out), dim=1)

        mu = self.mu(out)
        log_var = self.logvar(out)

        return mu, log_var



###-------------------------------------------------------------###
##                     ONTO DECODER CLASS                        ##
###-------------------------------------------------------------###


class OntoDecoder(nn.Module):
    """
    This class constructs an ontology structured Decoder module.
  
    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list of tuples that define in and out for each layer
    mask_list
        matrix for each layer transition, that determines which weights to zero out
    latent_dim
        latent dimension
    batches
        batch information of samples if available
    one_hot
        one-hot encoder for batch information if available
    neuronnum
        number of neurons to use per term
    """ 

    def __init__(self, in_features, layer_dims, mask_list, latent_dim, n_batch, one_hot=None, neuronnum=3):
        super(OntoDecoder, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.layer_dims = np.hstack([layer_dims[:-1] * neuronnum, layer_dims[-1]])
        self.layer_shapes = [(np.sum(self.layer_dims[:i+1]), self.layer_dims[i+1]) for i in range(len(self.layer_dims)-1)]
        self.masks = []
        for m in mask_list[0:-1]:
            m = m.repeat_interleave(neuronnum, dim=0)
            m = m.repeat_interleave(neuronnum, dim=1)
            self.masks.append(m.to(self.device))
        self.masks.append(mask_list[-1].repeat_interleave(neuronnum, dim=1).to(self.device))
        self.latent_dim = latent_dim
        self.n_batch = n_batch
        self.one_hot = one_hot.to(self.device) if one_hot is not None else one_hot

        # set batch information
        if self.n_batch > 0:
            self.layer_shapes[-1] = (self.layer_shapes[-1][0] + self.n_batch, self.layer_shapes[-1][1]) # concatenate batch dim to pathway activity dim
            self.masks[-1] = torch.hstack((self.masks[-1], torch.ones(self.masks[-1].shape[0], self.n_batch).to(self.device))) # concatenate batch vectors to masks (set to 1)


        # Decoder
        self.decoder = nn.ModuleList(

            [self.build_block(x[0], x[1]) for x in self.layer_shapes[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_shapes[-1][0], self.in_features)
                )
            ]
            ).to(self.device)
        
        # apply masks to zero out weights of non-existent connections
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = torch.mul(self.decoder[i][0].weight.data, self.masks[i])

        # make all weights in decoder positive
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = self.decoder[i][0].weight.data.clamp(0)

    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs)
        )

    def forward(self, z, batch=None):

        # decoding
        out = z

        for layer in self.decoder[:-1]:
            c = layer(out)
            out = torch.cat((c, out), dim=1)

        # attach batch info
        if batch is not None:
            if next(self.parameters()).is_cuda:
                out = torch.hstack((out, self.one_hot[batch]))
            else:
                out = torch.hstack((out, self.one_hot.to('cpu')[batch]))
        if batch is None and self.n_batch > 0:
            if next(self.parameters()).is_cuda:
                out = torch.hstack((out, torch.zeros((out.shape[0])).repeat(self.n_batch,1).T.to(self.device)))
            else:
                out = torch.hstack((out, torch.zeros((out.shape[0])).repeat(self.n_batch,1).T.to('cpu')))
        
        # final decoding layer
        reconstruction = self.decoder[-1](out)
        
        return reconstruction

