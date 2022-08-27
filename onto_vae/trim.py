import os
import contextlib
import io
import sys
import pandas as pd
import numpy as np
import itertools
import scanpy
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
import json
import pickle
from onto_vae.utils import *


class ontoobj():
    """
    This class is used for importing and processing an obo file and returning 
    all the neccessary files for training an OntoVAE model:
    - the DAG as dict
    - DAG annotation file
    - Ontology genes file
    - decoder masks
    - Wang semantic similarities

    Parameters
    -------------
    obo
        Path to the obo file
    gene_annot
        gene_annot
        Path two a tab-separated 2-column text file
        Gene1   Term1
        Gene1   Term211
        ...
    working_dir
        working directory where to store output files
    prefix
        prefix for output files, e.g. the ontology used
    """

    def __init__(self, obo, gene_annot, working_dir, prefix):
        super(ontoobj, self).__init__()

        self.dag = get_godag(obo, optional_attrs={'relationship'}, prt=None)
        self.gene_annot = pd.read_csv(gene_annot, sep="\t", header=None)
        self.gene_annot.columns = ['Gene', 'ID']
        self.working_dir = working_dir
        self.prefix = prefix

    def dag_annot(self, **kwargs):

        """
        This function takes in a dag object imported by goatools package and returns
        an annotation pandas dataframe where each row is one term containing the following columns:
        'ID': The ID of the DAG term
        'Name': The name 
        'depth': the depth (longest distance to a root node)
        'children': number of children of the term
        'parents': number of parents of the term
        The pandas dataframe is sorted by 'depth' and 'ID'
        
        Parameters
        ----------
        obo
            Path to the obo file
        **kwargs
            to pass if ids should be filtered, 
            {'id':'biological_process'}
        """
  
        # parse obo file and create list of term ids
        term_ids = list(set([vars(self.dag[term_id])['id'] for term_id in list(self.dag.keys())]))

        # if an id type was passed in kwargs, filter based on that
        if 'id' in kwargs.keys():
            term_idx = [vars(self.dag[term_id])['namespace'] == kwargs['id'] for term_id in term_ids]
            valid_ids = [N for i,N in enumerate(term_ids) if term_idx[i] == True]
            term_ids = valid_ids
            self.gene_annot = self.gene_annot[self.gene_annot.ID.isin(term_ids)]
        
        # extract information for annot file
        terms = [vars(self.dag[term_id])['name'] for term_id in term_ids]
        depths = [vars(self.dag[term_id])['depth'] for term_id in term_ids]
        num_children = [len(vars(self.dag[term_id])['children']) for term_id in term_ids]
        num_parents = [len(vars(self.dag[term_id])['parents']) for term_id in term_ids]

        # create annotation pandas dataframe
        annot = pd.DataFrame({'ID': term_ids,
                        'Name': terms,
                        'depth': depths,
                        'children': num_children,
                        'parents': num_parents})
        annot = annot.sort_values(['depth', 'ID'])
        return annot


    def create_dag_dict_files(self, **kwargs):

        """
        This function creates the following files

        prefix_graph.json in graph subfolder
            dictionary format of the dag with mapping children (keys) -> parents (values)

        prefix_annot.csv in annot subfolder
            annotation file for the dict with following columns:
            'ID': The ID of the DAG term
            'Name': The name 
            'depth': the depth (longest distance to a root node)
            'children': number of children of the term
            'parents': number of parents of the term
            'descendants': number of descendant terms
            'desc_genes': number of genes annotated to term and all its descendants
            'genes': number of genes directly annotated to term
        
        prefix_genes.txt in genes subfolder
            one-column text file containing the ontology genes alphabetically sorted

        **kwargs
            to pass if ids should be filtered, 
            id = 'biological_process'

        Terms with 0 descendant genes are removed!
        """

        # create saving subdirectories
        if not os.path.exists(self.working_dir + '/graph'):
            os.mkdir(self.working_dir + '/graph')
        if not os.path.exists(self.working_dir + '/annot'):
            os.mkdir(self.working_dir + '/annot')
        if not os.path.exists(self.working_dir + '/genes'):
            os.mkdir(self.working_dir + '/genes')

        # create initial annot file
        if 'id' in kwargs.keys():
            annot = self.dag_annot(id=kwargs['id'])
        else:
            annot = self.dag_annot()

        # convert gene annot file to dictionary
        gene_term_dict = {a: b["ID"].tolist() for a,b in self.gene_annot.groupby("Gene")}

        # convert the dag to a dictionary
        term_term_dict = {term_id: [x for x in vars(self.dag[term_id])['_parents'] if x in annot.ID.tolist()] for term_id in annot[annot.depth > 0].ID.tolist()}

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
        term_dict = {term_id: [x for x in vars(self.dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
        term_dict.update(gene_term_dict)

        # save the DAG dictionary as json file
        with open(self.working_dir + '/graph/' + self.prefix + '_graph.json', 'w') as jfile:
            json.dump(term_dict, jfile, sort_keys=True, indent=4)
        print('Ontology dictionary has been saved.')
        self.onto_dict = term_dict

        # update the annotation file

        # number of annotated genes
        term_size = self.gene_annot['ID'].value_counts().reset_index()
        term_size.columns = ['ID', 'genes']
        annot_updated = pd.merge(annot_updated, term_size, how='left', on='ID')
        annot_updated['genes'] = annot_updated['genes'].fillna(0)

        # recalculate number of children
        all_parents = list(term_dict.values())
        all_parents = [item for sublist in all_parents for item in sublist]
        refined_children = [all_parents.count(pid) - annot_updated[annot_updated.ID == pid].genes.values[0] for pid in annot_updated.ID.tolist()]
        annot_updated['children'] = refined_children

        # recalculate number of descendants
        term_dict = {term_id: [x for x in vars(self.dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
        term_dict_rev = reverse_graph(term_dict)
        num_desc = []
        for term in annot_updated.ID.tolist():
            desc = get_descendants(term_dict_rev, term)
            num_desc.append(len(set(desc)) - 1)
        annot_updated['descendants'] = num_desc 

        # save annot
        annot_updated.to_csv(self.working_dir + '/annot/' + self.prefix + '_annot.csv', index=False, sep=";")
        print('Annotation file has been saved.')
        self.onto_annot = annot_updated

        # save the genes as one-column text file
        genes = sorted(list(set(self.gene_annot.Gene.tolist())))
        pd.Series(genes).to_csv(self.working_dir + '/genes/' + self.prefix + '_genes.txt', header=False, index=False)
        print('Ontology genes have been saved.')
        self.onto_genes = genes


    def create_trim_dag_files(self, top_thresh=1000, bottom_thresh=30):

        """
        This function trims the DAG based on user-defined thresholds and saves trimmed versions 
        of the graph, annot and genes files.
            
        Output
        ----------
        prefix_trimmed_graph.json in subfolder graph
        prefix_trimmed_annot.csv in subfolder annot
        prefix_trimmed_genes.txt in subfolder genes

        Parameters
        ----------
        top_thresh
            top threshold for trimming: terms with > desc_genes will be pruned
        bottom_thresh
            bottom_threshold for trimming: terms with < desc_genes will be pruned and
            their genes will be transferred to their parents
        """

        # check if neccesary files exist and load them 
        if os.path.isfile(self.working_dir + '/graph/' + self.prefix + '_graph.json'):
            with open(self.working_dir + '/graph/' + self.prefix + '_graph.json') as jfile:
                onto_dict = json.load(jfile)
        else:
            sys.exit('graph file missing, create_dag_dict_files function needs to be run first!')
        
        if os.path.isfile(self.working_dir + '/annot/' + self.prefix + '_annot.csv'):
            onto_annot = pd.read_csv(self.working_dir + '/annot/' + self.prefix + '_annot.csv', sep=";")
        else:
            sys.exit('annot file missing, create_dag_dict_files function needs to be run first!')

        if os.path.isfile(self.working_dir + '/genes/' + self.prefix + '_genes.txt'):
            onto_genes = pd.read_csv(self.working_dir + '/genes/' + self.prefix + '_genes.txt', header=None)
        else:
            sys.exit('genes file missing, create_dag_dict_files function needs to be run first!')

        # get terms for trimming
        top_terms = onto_annot[onto_annot.desc_genes > top_thresh].ID.tolist()
        bottom_terms = onto_annot[onto_annot.desc_genes < bottom_thresh].ID.tolist()[::-1]

        # trim the DAG
        with contextlib.redirect_stdout(io.StringIO()):
            term_dict_ttrim = trim_DAG_top(onto_dict, onto_annot.ID.tolist(), top_terms)
        with contextlib.redirect_stdout(io.StringIO()):
            term_dict_trim = trim_DAG_bottom(term_dict_ttrim, onto_annot.ID.tolist(), bottom_terms)

        ### ANNOTATION FILE UPDATE ###

        # adjust the annotation file
        new_annot = onto_annot[onto_annot.ID.isin(top_terms + bottom_terms) == False].reset_index(drop=True)

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
        new_annot = new_annot.sort_values(['depth', 'ID'])

        ### OBJECT SAVING ###

        # trimmed DAG
        with open(self.working_dir + '/graph/' + self.prefix + '_trimmed_graph.json', 'w') as jfile:
            json.dump(term_dict_trim, jfile, sort_keys=True, indent=4)
        print('Trimmed Ontology dictionary has been saved.')
        self.onto_dict_trimmed = term_dict_trim

        # trimmed desc genes
        with open(self.working_dir + '/graph/' + self.prefix + '_trimmed_desc_genes.json', 'w') as jfile:
            json.dump(desc_genes, jfile, sort_keys=True, indent=4)

        # trimmed annot
        new_annot.to_csv(self.working_dir + '/annot/' + self.prefix + '_trimmed_annot.csv', index=False, sep=";")
        print('Trimmed Annotation file has been saved.')
        self.onto_annot_trimmed = new_annot

        # trimmed genes
        trim_genes = sorted(list(gene_trim.keys()))
        pd.Series(trim_genes).to_csv(self.working_dir + '/genes/' + self.prefix + '_trimmed_genes.txt', index=False, header=False)
        print('Trimmed Ontology genes have been saved.')
        self.onto_genes_trimmed = trim_genes


    def create_model_input(self, trimmed=True):

        """
        This function generates the masks for the Onto VAE model.

        Output
        ----------
        prefix_(trimmed)_decoder_masks.pickle
            
        Parameters
        ----------
        trimmed
            whether masks should be generated for trimmed data
            default: True
        """

        # check if neccesary files exist and load them 
        if trimmed==True:
            add_prefix = '_trimmed'
            if os.path.isfile(self.working_dir + '/graph/' + self.prefix + '_trimmed_graph.json'):
                with open(self.working_dir + '/graph/' + self.prefix + '_trimmed_graph.json') as jfile:
                    onto_dict = json.load(jfile)
            else:
                sys.exit('trimmed graph file missing, create_trim_dag_files function needs to be run first!')
            
            if os.path.isfile(self.working_dir + '/annot/' + self.prefix + '_trimmed_annot.csv'):
                onto_annot = pd.read_csv(self.working_dir + '/annot/' + self.prefix + '_trimmed_annot.csv', sep=";")
            else:
                sys.exit('trimmed annot file missing, create_trim_dag_files function needs to be run first!')

            if os.path.isfile(self.working_dir + '/genes/' + self.prefix + '_trimmed_genes.txt'):
                onto_genes = pd.read_csv(self.working_dir + '/genes/' + self.prefix + '_trimmed_genes.txt', header=None)
            else:
                sys.exit('trimmed genes file missing, create_trim_dag_files function needs to be run first!')
        else:
            add_prefix = ''
            if os.path.isfile(self.working_dir + '/graph/' + self.prefix + '_graph.json'):
                with open(self.working_dir + '/graph/' + self.prefix + '_graph.json') as jfile:
                    onto_dict = json.load(jfile)
            else:
                sys.exit('graph file missing, create_dag_dict_files function needs to be run first!')
            
            if os.path.isfile(self.working_dir + '/annot/' + self.prefix + '_annot.csv'):
                onto_annot = pd.read_csv(self.working_dir + '/annot/' + self.prefix + '_annot.csv', sep=";")
            else:
                sys.exit('annot file missing, create_dag_dict_files function needs to be run first!')

            if os.path.isfile(self.working_dir + '/genes/' + self.prefix + '_genes.txt'):
                onto_genes = pd.read_csv(self.working_dir + '/genes/' + self.prefix + '_genes.txt', header=None)
            else:
                sys.exit('genes file missing, create_dag_dict_files function needs to be run first!')

        # create subdirectory
        if not os.path.exists(self.working_dir + '/masks'):
            os.mkdir(self.working_dir + '/masks')

        # get all possible depth combos
        depth = onto_annot.loc[:,['ID', 'depth']]
        gene_depth = pd.DataFrame({'ID': onto_genes.iloc[:,0].tolist(), 'depth': np.max(depth.depth)+1})
        depth = pd.concat([depth.reset_index(drop=True), gene_depth], axis=0)
        depth_combos = list(itertools.combinations(list(set(depth['depth'])), 2))

        # create binary matrix for all possible depth combos
        bin_mat_list = [create_binary_matrix(depth, onto_dict, p[1], p[0]) for p in depth_combos]

        # generate masks for the decoder network
        levels = ['Level' + str(d) for d in list(set(depth['depth'].tolist()))]
        mask_cols = [list(levels)[0:i+1][::-1] for i in range(len(levels)-1)]
        mask_rows = levels[1:]

        idx = [[mat.columns.name in mask_cols[i] and mat.index.name == mask_rows[i] for mat in bin_mat_list] for i in range(len(mask_rows))]
        masks = [np.array(pd.concat([N for i,N in enumerate(bin_mat_list) if j[i] == True][::-1], axis=1)) for j in idx]

        # save new mask file
        with open(self.working_dir + '/masks/' + self.prefix + add_prefix + '_decoder_masks.pickle', 'wb') as f:
            pickle.dump(masks, f) 
        print('Decoder masks have been saved.')


    def compute_wsem_sim(self, trimmed=True):

        """
        This function takes an obo file and an ontology annot file and returns
        a 2D numpy array with Wang semantic similarities between the IDs of the annot file.
        
        Parameters
        ----------
        trimmed
            if trimmed version of the dag should be used
            default: True 
        """

        # check if neccesary files exist and load them 
        if trimmed==True:
            add_prefix = '_trimmed'
            if os.path.isfile(self.working_dir + '/annot/' + self.prefix + '_trimmed_annot.csv'):
                onto_annot = pd.read_csv(self.working_dir + '/annot/' + self.prefix + '_trimmed_annot.csv', sep=";")
            else:
                sys.exit('trimmed annot file missing, create_trim_dag_files function needs to be run first!')
        else:
            add_prefix = ''
            if os.path.isfile(self.working_dir + '/annot/' + self.prefix + '_annot.csv'):
                onto_annot = pd.read_csv(self.working_dir + '/annot/' + self.prefix + '_annot.csv', sep=";")
            else:
                sys.exit('annot file missing, create_dag_dict_files function needs to be run first!')

        # create subdirectory
        if not os.path.exists(self.working_dir + '/sem_sim'):
            os.mkdir(self.working_dir + '/sem_sim')
        add_prefix = ''

        # get list of IDs
        ids = onto_annot['ID'].tolist()

        # compute wang semantic similarities
        wang = SsWang(ids, self.dag)
        wsem_sim = [[wang.get_sim(id1, id2) for id2 in ids] for id1 in ids]
        wsem_sim = np.array(wsem_sim)

        np.save(self.working_dir + '/sem_sim/' + self.prefix + add_prefix + '_wsem_sim.npy', wsem_sim)
        print('Matrix with Wang Semantic Similarities has been saved.')


    def match_dataset(self, expr_path, name):

        """
        This function takes a dataset and matches the features to the features of the preprocessed ontology.
        
        Output
        ----------
        (name)_(prefix)_trimmed_expr.npy: a numpy 2D array containing the matched dataset

        Parameters
        ----------
        expr_path: Path to the dataset to be matched, can be either:
              - a h5ad file (extension .h5ad)
              - a file with extension .csv (separated by ',') or with extension .txt (separated by '\t'), 
                with features in rows and samples in columns
        name: name to be used for saving the matched dataset
        """

        # check if ontology has been trimmed and import the genes file

        if os.path.isfile(self.working_dir + '/genes/' + self.prefix + '_trimmed_genes.txt'):
            onto_genes = pd.read_csv(self.working_dir + '/genes/' + self.prefix + '_trimmed_genes.txt', header=None)
        else:
            sys.exit('trimmed genes file missing, create_trim_dag_files function needs to be run first!')

        # check file extension of dataset to be matched

        basename = os.path.basename(expr_path)
        ext = basename.split('.')[1]

        if ext == 'csv':
            expr = pd.read_csv(expr_path, sep=",", index_col=0)
        elif ext == 'txt':
            expr = pd.read_csv(expr_path, sep="\t", index_col=0)
        elif ext == 'h5ad':
            # read in with scanpy
            data = scanpy.read_h5ad(expr_path)
            sample_annot = data.obs
            genes = data.var.gene_symbol.reset_index(drop=True)
            expr = data.X.todense()
            # convert to pandas dataframe
            expr = pd.DataFrame(expr.T)
            expr.index = genes
            expr.columns = sample_annot.index
        else:
            sys.exit('File extension not supported.')

        # merge data with ontology genes and save
        onto_genes.index = onto_genes.iloc[:,0]
        merged_expr = onto_genes.join(expr).fillna(0).drop(0, axis=1).T
        np.save(os.path.dirname(expr_path) + '/' + name + '_' + self.prefix + '_trimmed_expr.npy', merged_expr.to_numpy())
        


