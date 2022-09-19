#!/usr/bin/env python3

# Modules used by OntoVAE

import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

###-------------------------------------------------------------###
##                       ENCODER CLASS                           ##
###-------------------------------------------------------------###

class Encoder(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.

    Parameters
    -------------
    in_features: # of features that are used as input
    layer_dims: list giving the dimensions of the hidden layers
    latent_dim: latent dimension
    drop: dropout rate, default is 0
    z_drop: dropout rate for latent space, default is 0.5
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
        #c = c.view(-1, 2, self.latent_dim)

        # get 'mu' and 'log-var'
        #mu = c[:, 0, :]
        #log_var = c[:, 1, :]

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
    -------------
    in_features: # of features that are used as input
    layer_dims: list giving the dimensions of the hidden layers
    latent_dim: latent dimension, default is 128
    dr: dropout rate, default is 0
    """

    def __init__(self, in_features, latent_dim, layer_dims=[512], drop=0):
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
    This class constructs a Encoder module that is structured like an ontology and following a DAG.

    Parameters
    --------------
    in_features: # of features that are used as input
    layer_dims: list of tuples that define in and out for each layer
    mask_list: matrix for each layer transition, that determines which weights to zero out
    drop: dropout rate, default is 0
    z_drop: dropout rate for latent space, default is 0.5
    """ 

    def __init__(self, in_features, layer_dims, mask_list, drop=0, z_drop=0.5):
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

        #c = c.view(-1, 2, self.latent_dim)

        # get 'mu' and 'log-var'
        #mu = lat[:, 0, :]
        #log_var = lat[:, 1, :]

        mu = self.mu(out)
        log_var = self.logvar(out)

        return mu, log_var



###-------------------------------------------------------------###
##                     ONTO DECODER CLASS                        ##
###-------------------------------------------------------------###


class OntoDecoder(nn.Module):
    """
    This class constructs a Decoder module that is structured like an ontology and following a DAG.
  
    Parameters
    ---------------
    in_features: # of features that are used as input
    study_num: # of different studys that the samples belong to
    layer_dims: list of tuples that define in and out for each layer
    mask_list: matrix for each layer transition, that determines which weights to zero out
    latent_dim: latent dimension
    drop: dropout rate, default is 0
    """ 

    def __init__(self, in_features, layer_dims, mask_list, latent_dim, neuronnum=1, drop=0):
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
        self.drop = drop

        # Decoder
        self.decoder = nn.ModuleList(

            [self.build_block(x[0], x[1]) for x in self.layer_shapes[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_shapes[-1][0], self.in_features)#,
                    #nn.Sigmoid()
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
            nn.Linear(ins, outs)#,
            #nn.Dropout(p=self.drop),
            #nn.Sigmoid()
        )

    def forward(self, z):

        # decoding
        out = z

        for layer in self.decoder[:-1]:
            c = layer(out)
            out = torch.cat((c, out), dim=1)
        reconstruction = self.decoder[-1](out)
        
        return reconstruction

