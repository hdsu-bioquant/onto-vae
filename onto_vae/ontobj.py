import os
import contextlib
import io
import sys
import pandas as pd
import numpy as np
import itertools
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
import matplotlib.pyplot as plt 
import seaborn as sns
import colorcet as cc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from onto_vae.utils import *


class Ontobj():
    """
    This class functions as a container for a preprocessed ontology and matched datasets.
    An instance of the class Ontobj is needed by the OntoVAE class to train OntoVAE models.

    Predefined slots:
    annot_base: contains annotation files for ontology with the following columns
            'ID': The ID of the DAG term
            'Name': The name 
            'depth': the depth (longest distance to a root node)
            'children': number of children of the term
            'parents': number of parents of the term
            'descendants': number of descendant terms
            'desc_genes': number of genes annotated to term and all its descendants
            'genes': number of genes directly annotated to term
    genes_base: contains genes that can be mapped to ontology in alphabetical order
    graph_base: a dictionary with ontology relationships (children -> parents)
    annot, genes and graph can contain different trimmed versions
    desc_genes: a dictionary with all descendant genes (terms -> descendant genes)
    sem_sim: semantic similarities for all genes of one of the elements in the genes slot
    data: a dictionary containing 2d numpy arrays with expression values, one for each dataset matched to the ontology

    Parameters
    ----------
    description
        to identify the object, used ontology or gene identifiers can be specified here, for example 'GO' or 'HPO' or 'GO_BP'
    """

    __slots__=(
        'description', 
        'identifiers', 
        'annot_base', 
        'genes_base', 
        'graph_base', 
        'annot', 
        'genes', 
        'graph', 
        'desc_genes', 
        'masks', 
        'sem_sim', 
        'data'
        )

    def __init__(self, description):
        super(Ontobj, self).__init__()

        self.description = description
        self.identifiers = None
        self.annot_base = None
        self.genes_base = None
        self.graph_base = None
        self.annot = {}
        self.genes = {}
        self.graph = {}
        self.desc_genes = {}
        self.masks = {}
        self.sem_sim = {}
        self.data = {}

    def _dag_annot(self, dag, gene_annot, filter_id=None):

        """
        Creates annotation dataframe from imported obo file.
        
        Parameters
        ----------
        dag
            a dag parsed from an obo file
        gene_annot
            pandas dataframe containing gene -> term annotation
        filter_id
            to pass if ids should be filtered, e.g.
            filter_id = 'biological_process'

        Returns
        -------
        A pandas dataframe with ontology annotation, sorted by depth and ID.
        Each row is one term, the columns are:
            'ID': The ID of the ontology term
            'Name': The name of the ontology term
            'depth': the depth (longest distance to a root node)
            'children': number of children of the term
            'parents': number of parents of the term
        """
  
        # parse obo file and create list of term ids
        term_ids = list(set([vars(dag[term_id])['id'] for term_id in list(dag.keys())]))

        # if filter_id was specified, filter the ontology terms
        if filter_id is not None:
            term_idx = [vars(dag[term_id])['namespace'] == filter_id for term_id in term_ids]
            valid_ids = [N for i,N in enumerate(term_ids) if term_idx[i] == True]
            term_ids = valid_ids
            gene_annot = gene_annot[gene_annot.ID.isin(term_ids)]
        
        # extract information for annot file
        terms = [vars(dag[term_id])['name'] for term_id in term_ids]
        depths = [vars(dag[term_id])['depth'] for term_id in term_ids]
        num_children = [len(vars(dag[term_id])['children']) for term_id in term_ids]
        num_parents = [len(vars(dag[term_id])['parents']) for term_id in term_ids]

        # create annotation pandas dataframe
        annot = pd.DataFrame({'ID': term_ids,
                        'Name': terms,
                        'depth': depths,
                        'children': num_children,
                        'parents': num_parents})
        annot = annot.sort_values(['depth', 'ID'])
        return annot


    def initialize_dag(self, obo, gene_annot, filter_id = None):

        """
        Dag is initialized from obo file and annotation.
        The slots annot_base, genes_base, and graph_base are filled.

        Parameters
        -------------
        obo
            Path to the obo file

        gene_annot
            gene_annot
            Path two a tab-separated 2-column text file
            Gene1   Term1
            Gene1   Term2
            ...

        filter_id
            to pass if ids should be filtered, e.g.
            filter_id = 'biological_process'
        """

        # load obo file and gene -> term mapping file
        dag = get_godag(obo, optional_attrs={'relationship'}, prt=None)
        gene_annot = pd.read_csv(gene_annot, sep="\t", header=None)
        gene_annot.columns = ['Gene', 'ID']

        self.identifiers = 'Ensembl' if 'ENS' in gene_annot.iloc[0,0] else 'HGNC'

        # create initial annot file
        if filter_id is not None:
            annot = self._dag_annot(dag, gene_annot, filter_id=filter_id)
        else:
            annot = self._dag_annot(dag, gene_annot)

        # convert gene annot file to dictionary
        gene_term_dict = {a: b["ID"].tolist() for a,b in gene_annot.groupby("Gene")}

        # convert the dag to a dictionary
        term_term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot.ID.tolist()] for term_id in annot[annot.depth > 0].ID.tolist()}

        # reverse the DAG to be able to count descendants and descendant genes
        gene_dict_rev = reverse_graph(gene_term_dict)
        term_dict_rev = reverse_graph(term_term_dict)

        # count descendants and descendant genes and add to annot
        num_desc = []
        num_genes = []

        for term in annot.ID.tolist():
            desc = get_descendants(term_dict_rev, term)
            num_desc.append(len(set(desc)) - 1)
            genes = get_descendant_genes(gene_dict_rev, desc)
            num_genes.append(len(set(genes)))
        
        annot['descendants'] = num_desc
        annot['desc_genes'] = num_genes

        # remove terms that don't have any descendant genes
        annot_updated = annot[annot.desc_genes > 0]
        annot_updated = annot_updated.sort_values(['depth', 'ID']).reset_index(drop=True)

        # update the dag dict using only the good IDs
        term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
        term_dict.update(gene_term_dict)

        # update the annotation file

        # number of annotated genes
        term_size = gene_annot['ID'].value_counts().reset_index()
        term_size.columns = ['ID', 'genes']
        annot_updated = pd.merge(annot_updated, term_size, how='left', on='ID')
        annot_updated['genes'] = annot_updated['genes'].fillna(0)

        # recalculate number of children
        all_parents = list(term_dict.values())
        all_parents = [item for sublist in all_parents for item in sublist]
        refined_children = [all_parents.count(pid) - annot_updated[annot_updated.ID == pid].genes.values[0] for pid in annot_updated.ID.tolist()]
        annot_updated['children'] = refined_children

        # recalculate number of descendants
        term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
        term_dict_rev = reverse_graph(term_dict)
        num_desc = []
        for term in annot_updated.ID.tolist():
            desc = get_descendants(term_dict_rev, term)
            num_desc.append(len(set(desc)) - 1)
        annot_updated['descendants'] = num_desc 

        # fill the basic slots
        self.annot_base = annot_updated
        self.genes_base = sorted(list(set(gene_annot.Gene.tolist())))
        term_dict.update(gene_term_dict)
        self.graph_base = term_dict


    def trim_dag(self, top_thresh=1000, bottom_thresh=30):

        """
        DAG is trimmed based on user-defined thresholds.
        Trimmed version is saved in the graph, annot and genes slots.

        Parameters
        ----------
        top_thresh
            top threshold for trimming: terms with > desc_genes will be pruned
        bottom_thresh
            bottom_threshold for trimming: terms with < desc_genes will be pruned and
            their genes will be transferred to their parents
        """

        # check if base versions of files exits
        if self.graph_base is None:
            sys.exit('intial graph has not been created, initialize_dag function needs to be run first!')
        else:
            graph_base = self.graph_base.copy()

        if self.annot_base is None:
            sys.exit('initial annot has not been created, initialize_dag function needs to be run first!')
        else:
            annot_base = self.annot_base.copy()

        if self.genes_base is None:
            sys.exit('initial genes has not been created, initialize_dag function needs to be run first!')
        else:
            genes_base = self.genes_base.copy()

        # get terms for trimming
        top_terms = annot_base[annot_base.desc_genes > top_thresh].ID.tolist()
        bottom_terms = annot_base[annot_base.desc_genes < bottom_thresh].ID.tolist()[::-1]

        # trim the DAG
        with contextlib.redirect_stdout(io.StringIO()):
            term_dict_ttrim = trim_DAG_top(graph_base, annot_base.ID.tolist(), top_terms)
        with contextlib.redirect_stdout(io.StringIO()):
            term_dict_trim = trim_DAG_bottom(term_dict_ttrim, annot_base.ID.tolist(), bottom_terms)

        ### ANNOTATION FILE UPDATE ###

        # adjust the annotation file
        new_annot = annot_base[annot_base.ID.isin(top_terms + bottom_terms) == False].reset_index(drop=True)

        # split the DAG
        term_trim = {key: term_dict_trim[key] for key in list(term_dict_trim.keys()) if key in new_annot.ID.tolist()}
        gene_trim = {key: term_dict_trim[key] for key in list(term_dict_trim.keys()) if key not in new_annot.ID.tolist()}  

        # reverse the separate DAGs
        term_trim_rev = reverse_graph(term_trim)
        gene_trim_rev = reverse_graph(gene_trim)

        # calculate new children, parent and gene numbers
        new_children = [len(term_trim_rev[term]) if term in list(term_trim_rev.keys()) else 0 for term in new_annot.ID.tolist()]
        new_parents = [len(term_trim[term]) if term in list(term_trim.keys()) else 0 for term in new_annot.ID.tolist()]
        new_genes = [len(gene_trim_rev[term]) if term in list(gene_trim_rev.keys()) else 0 for term in new_annot.ID.tolist()]

        # calculate new descendants and descendant genes
        num_desc = []
        num_genes = []

        desc_genes = {}

        for term in new_annot.ID.tolist():
            desc = get_descendants(term_trim_rev, term)
            num_desc.append(len(set(desc)) - 1)
            genes = set(get_descendant_genes(gene_trim_rev, desc))
            desc_genes[term] = list(genes)
            num_genes.append(len(genes))
        
        # update the annot file
        new_annot['children'] = new_children
        new_annot['parents'] = new_parents
        new_annot['genes'] = new_genes
        new_annot['descendants'] = num_desc
        new_annot['desc_genes'] = num_genes

        # set the depth of all terms with 0 parents to 0
        new_annot.loc[new_annot.parents == 0, 'depth'] = 0

        # adjust depth of other terms
        min_depth = np.min(new_annot['depth'][new_annot['depth'] != 0])

        def adjust_depth(row):
            if row['depth'] > 0:
                return row['depth'] - (min_depth - 1)
            else:
                return 0
        
        new_annot['depth'] = new_annot.apply(lambda row: adjust_depth(row), axis=1)
        new_annot = new_annot.sort_values(['depth', 'ID']).reset_index(drop=True)

        # save trimming results in respective slots
        self.annot[str(top_thresh) + '_' + str(bottom_thresh)] = new_annot
        self.graph[str(top_thresh) + '_' + str(bottom_thresh)] = term_dict_trim
        self.genes[str(top_thresh) + '_' + str(bottom_thresh)] = sorted(list(gene_trim.keys()))
        self.desc_genes[str(top_thresh) + '_' + str(bottom_thresh)] = desc_genes


    def create_masks(self, top_thresh=1000, bottom_thresh=30, module='decoder'):

        """
        Creation of masks to initialize the wiring in the latent space and decoder of OntoVAE.
            
        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        module
            decoder: Ontology will be located in decoder (default)
            encoder: Ontology will be located in encoder 
        """

        # check if neccesary objects exist
        if str(top_thresh) + '_' + str(bottom_thresh) not in self.graph.keys():
            sys.exit('trimmed graph with specified thresholds missing, trim_dag function needs to be run first!')
        else:
            onto_dict = self.graph[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        if str(top_thresh) + '_' + str(bottom_thresh) not in self.annot.keys():
            sys.exit('trimmed annot with specified thresholds missing, trim_dag function needs to be run first!')
        else:
            annot = self.annot[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        if str(top_thresh) + '_' + str(bottom_thresh) not in self.genes.keys():
            sys.exit('trimmed genes with specified thresholds missing, trim_dag function needs to be run first!')
        else:
            genes = self.genes[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        # check if mask slot for thresholds exists
        if str(top_thresh) + '_' + str(bottom_thresh) not in self.masks.keys():
            self.masks[str(top_thresh) + '_' + str(bottom_thresh)] = {}

        # get all possible depth combos
        depth = annot.loc[:,['ID', 'depth']]
        gene_depth = pd.DataFrame({'ID': genes, 'depth': np.max(depth.depth)+1})
        depth = pd.concat([depth.reset_index(drop=True), gene_depth], axis=0)
        depth_combos = list(itertools.combinations(list(set(depth['depth'])), 2))

        # create binary matrix for all possible depth combos
        bin_mat_list = [create_binary_matrix(depth, onto_dict, p[1], p[0]) for p in depth_combos]

        # generate masks for the ontology network
        if module == 'decoder':
            masks = self._decoder_masks(depth, bin_mat_list)
        else:
            masks = self._encoder_masks(depth, bin_mat_list)

        # store masks
        self.masks[str(top_thresh) + '_' + str(bottom_thresh)][module] = masks


    def _decoder_masks(self, depth, bin_mat_list):
        """
        Helper function to create binary masks for decoder
        """
        levels = ['Level' + str(d) for d in list(set(depth['depth'].tolist()))]
        mask_cols = [list(levels)[0:i+1][::-1] for i in range(len(levels)-1)]
        mask_rows = levels[1:]

        idx = [[mat.columns.name in mask_cols[i] and mat.index.name == mask_rows[i] for mat in bin_mat_list] for i in range(len(mask_rows))]
        masks = [np.array(pd.concat([N for i,N in enumerate(bin_mat_list) if j[i] == True][::-1], axis=1)) for j in idx]
        return masks

    def _encoder_masks(self, depth, bin_mat_list):
        """
        Helper function to create binary masks for encoder
        """
        levels = ['Level' + str(d) for d in list(set(depth['depth'].tolist()))][::-1]
        mask_cols = [list(levels)[0:i+1][::-1] for i in range(len(levels)-1)]
        mask_rows = levels[1:]

        idx = [[mat.index.name in mask_cols[i] and mat.columns.name == mask_rows[i] for mat in bin_mat_list] for i in range(len(mask_rows))]
        masks = [np.array(pd.concat([N.T for i,N in enumerate(bin_mat_list) if j[i] == True], axis=1)) for j in idx]
        return masks



    def compute_wsem_sim(self, obo, top_thresh=1000, bottom_thresh=30):

        """
        Wang semantic similarities between a list of ontology terms are computed.
        
        Parameters
        ----------
        obo
            Path to the obo file
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """

        # check if neccesary files exist and load them 
        if str(top_thresh) + '_' + str(bottom_thresh) not in self.annot.keys():
            sys.exit('trimmed annot with specified thresholds missing, trim_dag function needs to be run first!')
        else:
            annot = self.annot[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        # parse the DAG
        dag = get_godag(obo, optional_attrs={'relationship'}, prt=None)

        # get list of IDs
        ids = annot['ID'].tolist()

        # compute wang semantic similarities
        wang = SsWang(ids, dag)
        wsem_sim = [[wang.get_sim(id1, id2) for id2 in ids] for id1 in ids]
        wsem_sim = np.array(wsem_sim)

        # store results
        self.sem_sim[str(top_thresh) + '_' + str(bottom_thresh)] = wsem_sim



    def match_dataset(self, expr_data, name, top_thresh=1000, bottom_thresh=30):

        """
        A dataset is matched to a trimmed ontology and stored in the data slot.

        Parameters
        ----------
        expr_data
            dataset with unique gene names in rows and samples in columns, can be either:
            - a Pandas dataframe 
            - a path to the dataset file, can be either:
              - a file with extension .csv (separated by ',')
              - a file with extension .txt (separated by '\t') 

        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        name
            name to be used for identifying the matched dataset
        """

        # check if ontology has been trimmed and import the genes file

        if str(top_thresh) + '_' + str(bottom_thresh) not in self.genes.keys():
            sys.exit('trimmed genes with specified thresholds missing, trim_dag function needs to be run first!')
        else:
            genes = pd.DataFrame(self.genes[str(top_thresh) + '_' + str(bottom_thresh)])

        # check file extension of dataset to be matched
        if isinstance(expr_data, pd.DataFrame):
            expr = expr_data
        else:
            basename = os.path.basename(expr_data)
            ext = basename.split('.')[1]

            if ext == 'csv':
                expr = pd.read_csv(expr_data, sep=",", index_col=0)
            elif ext == 'txt':
                expr = pd.read_csv(expr_data, sep="\t", index_col=0)
            else:
                sys.exit('File extension not supported.')

        # merge data with ontology genes and save
        genes.index = genes.iloc[:,0]
        merged_expr = genes.join(expr).fillna(0).drop(0, axis=1).T

        if str(top_thresh) + '_' + str(bottom_thresh) not in self.data.keys():
            self.data[str(top_thresh) + '_' + str(bottom_thresh)] = {}

        self.data[str(top_thresh) + '_' + str(bottom_thresh)][name] = merged_expr.to_numpy()

    
    def extract_annot(self, top_thresh=1000, bottom_thresh=30):
        """
        Helper function to extract table from annot slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        return self.annot[str(top_thresh) + '_' + str(bottom_thresh)]

    def extract_genes(self, top_thresh=1000, bottom_thresh=30):
        """
        Helper function to extract list from genes slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        return self.genes[str(top_thresh) + '_' + str(bottom_thresh)]

    def extract_dataset(self, dataset, top_thresh=1000, bottom_thresh=30):
        """
        Helper function to extract dataset from data slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        return self.data[str(top_thresh) + '_' + str(bottom_thresh)][dataset]

    
    def add_dataset(self, dataset, name, top_thresh=1000, bottom_thresh=30):
        """
        Helper function to add matched dataset to data slot.

        Parameters
        ----------
        dataset
            2d numpy array of the dataset to be added 
        name
            name of the dataset
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        self.data[str(top_thresh) + '_' + str(bottom_thresh)][name] = dataset


    def remove_link(self, term, gene, top_thresh=1000, bottom_thresh=30):
        """
        Modifies the masks slot by removing the link between a gene and a term.

        Parameters
        ----------
        term
            id of the term
        gene
            the gene
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        onto_annot = self.extract_annot(top_thresh=top_thresh,
                                        bottom_thresh=bottom_thresh)
        genes = self.extract_genes(top_thresh=top_thresh,
                                        bottom_thresh=bottom_thresh)

        # retrieve indices to remove link
        # for the term, we need to work around, as terms in masks are sorted reversed (Depth 15 -> Depth 14 -> Depth 13 ...)
        term_depth = onto_annot[onto_annot.ID == term].depth.to_numpy()[0]
        depth_counts = onto_annot.depth.value_counts().sort_index(ascending=False)
        start_point = depth_counts[depth_counts.index > term_depth].sum()
        annot_sub = onto_annot[onto_annot.depth == term_depth]
        term_idx = annot_sub[annot_sub.ID == term].index.to_numpy()
        gene_idx = genes.index(gene)

        self.masks[str(top_thresh) + '_' + str(bottom_thresh)][-1][gene_idx, start_point + term_idx] = 0


    def plot_scatter(self, sample_annot, color_by, act, term1, term2, top_thresh=1000, bottom_thresh=30):
        """ 
       Creates a scatterplot of two pathway activities.

        Parameters
        ----------
        sample_annot
            sample annotation with samples in rows and covariates in columns, can be either:
            - a Pandas dataframe 
            - a path to the dataset file, can be either:
              - a file with extension .csv (separated by ',')
              - a file with extension .txt (separated by '\t')

        color_by
            the column of sample_annot to use for coloring
        act
            numpy array containing pathway activities
        term1
            ontology term on x-axis
        term2
            ontology term on y-axis
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        # import sample annotation
        if isinstance(sample_annot, pd.DataFrame):
            sample_annot = sample_annot
        else:
            basename = os.path.basename(sample_annot)
            ext = basename.split('.')[1]

            if ext == 'csv':
                sample_annot = pd.read_csv(sample_annot, sep=",", index_col=0)
            elif ext == 'txt':
                sample_annot = pd.read_csv(sample_annot, sep="\t", index_col=0)

        # create color dict
        categs = sample_annot.loc[:,color_by].unique().tolist()
        palette = sns.color_palette(cc.glasbey, n_colors=len(categs))
        color_dict = dict(zip(categs, palette))

        # extract ontology annot and get term indices
        onto_annot = self.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
        ind1 = onto_annot[onto_annot.Name == term1].index.to_numpy()
        ind2 = onto_annot[onto_annot.Name == term2].index.to_numpy()

        # make scatterplot
        fig, ax = plt.subplots(figsize=(10,7))
        sns.scatterplot(x=act[:,ind1].flatten(),
                        y=act[:,ind2].flatten(),
                        hue=sample_annot.loc[:,color_by],
                        palette=color_dict,
                        legend='full',
                        s=8,
                        rasterized=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel(term1)
        plt.ylabel(term2)
        plt.tight_layout()

    
    def wilcox_test(self, control, perturbed, direction='up', option='terms', top_thresh=1000, bottom_thresh=30):
        """ 
        Performs paired Wilcoxon test between activities and perturbed activities.

        Parameters
        ----------
        act
            numpy 2D array of pathway activities 
        perturbed_act
            numpy 2D array of perturbed pathway activities
        direction
            up: higher in perturbed
            down: lower in perturbed
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        option
            'terms' or 'genes'
        """
        # perform paired wilcoxon test over all terms
        alternative = 'greater' if direction == 'up' else 'less'
        wilcox = [stats.wilcoxon(perturbed[:,i], control[:,i], zero_method='zsplit', alternative=alternative) for i in range(control.shape[1])]
        stat = np.array([i[0] for i in wilcox])
        pvals = np.array([i[1] for i in wilcox])
        qvals = fdrcorrection(np.array(pvals))

        if option == 'terms':
            # extract ontology annot
            onto_annot = self.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)

            # create results dataframe 
            res = pd.DataFrame({'id': onto_annot.ID.tolist(),
                                'term': onto_annot.Name.tolist(),
                                'depth': onto_annot.depth.tolist(),
                                'stat': stat,
                                'pval' : pvals,
                                'qval': qvals[1]})
        
        else:
            # extract ontology genes
            onto_genes = self.extract_genes(top_thresh=top_thresh, bottom_thresh=bottom_thresh)

            # create results dataframe
            res = pd.DataFrame({'gene': onto_genes,
                                'stat': stat,
                                'pval' : pvals,
                                'qval': qvals[1]})

        res = res.sort_values('pval').reset_index(drop=True)
        return(res)
