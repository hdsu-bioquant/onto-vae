import pandas as pd
import itertools
import copy
import pkg_resources



###--------------------------------------
## DAG REVERSAL
###--------------------------------------

def reverse_graph(graph):
    reverse = {}
    for v in graph:
        for e in graph[v]:
            if e not in reverse:
                reverse[e] = []
            reverse[e].append(v)
    return reverse



###--------------------------------------
## DESCENDANT GENE COUNTING
###--------------------------------------

# function to iterate over DAG for a given term and get all children and children's children
# this function has to be run with a reversed DAG, with mapping parents -> children, not containing gene annot
def get_descendants(dag, term):

    descendants = []
    queue = []

    descendants.append(term)
    queue.append(term)

    while len(queue) > 0:
        node = queue.pop(0)
        if node in list(dag.keys()):
            children = dag[node]
            descendants.extend(children)
            queue.extend(children)
        else:
            pass

    return descendants

# function to, given a list of descendant terms, get all genes associated to them
# the function has to be run with reversed gene annot dict, with mapping terms -> annotated genes
def get_descendant_genes(dag, descendants):

    desc_dict = {key: dag[key] for key in list(dag.keys()) if key in descendants}
    genes = list(desc_dict.values())
    genes = [item for sublist in genes for item in sublist]
    return genes





###--------------------------------------
## ONTOLOGY DECODER MASKS
###--------------------------------------

# function to create binary matrices 

def create_binary_matrix(depth, dag, childnum, parentnum):
    ## depth is a dataframe containing all IDs (ontology IDs and genes) ['ID'] and their respective depth ['Depth'] in the graph
    ## dag is a dictionary, keys: all IDs, values: parents of respective ID
    ## childnum is a number indicating which depth level to use as child level
    ## parentnum is number is a number indicating which depth level to use as parent level
    
    ## output is a binary sparse pandas dataframe with 0s and 1s indicating if there is a relationship
    ## between any two elements from the two depth levels investigated

    # create two lists of IDs from two different levels (child and parent)
    children = depth.loc[depth['depth'] == childnum, 'ID'].tolist()
    parents = depth.loc[depth['depth'] == parentnum, 'ID'].tolist()

    # create a dataframe with all possible combinations of the two lists
    df = pd.DataFrame(list(itertools.product(children, parents)), columns=['Level' + str(childnum), 'Level' + str(parentnum)])

    # for each potential child-parent pair, check if there is a relationship and add this to the df
    interact = [1 if y in dag[x] else 0 for x, y in zip(df['Level' + str(childnum)], df['Level' + str(parentnum)])]
    df['interact'] = interact

    # change from long format to wide format
    df = df.pivot(index='Level' + str(childnum), columns='Level' + str(parentnum), values='interact')

    return(df)




###--------------------------------------
## DAG TRIMMING
###--------------------------------------

# DAG trimming from bottom given a list of terms

###-------------->  helper function to trim one term

def trim_term_bottom(term, term_term_dict, term_dict_rev, gene_dict_rev):
    """
    Input
    -----------------
    term: the term to be trimmed off
    term_term_dict: mapping children -> parents excluding the genes
    term_dict_rev: parents(terms) -> children(terms)
    gene_dict_rev: parents(terms) -> children(genes)

    Output
    -----------------
    This function is changing the term_dict_rev and the gene_dict_rev variables
    """

    # check if term has parents (depth0 won't have)
    if term in list(term_term_dict.keys()):
        parents = copy.deepcopy(term_term_dict[term])
    else:
        parents = []

    # iterate over parents and remove the term from their children
    # also add the genes of the term to the genes of its parents
    if len(parents) > 0:
        for p in parents:
            term_dict_rev[p].remove(term)
            if p not in list(gene_dict_rev.keys()):
                gene_dict_rev[p] = []
            gene_dict_rev[p].extend(gene_dict_rev[term])
            gene_dict_rev[p] = list(set(gene_dict_rev[p])) # remove eventual duplicates

    # remove the term -> genes and term -> term entries from the dicts
    del gene_dict_rev[term]
    if term in list(term_dict_rev.keys()):
        del term_dict_rev[term]



###--------------> function to trim the DAG

def trim_DAG_bottom(DAG, all_terms, trim_terms):
    """
    Input
    -----------------
    DAG: the DAG to be trimmed
    all_terms: all ontology terms 
    trim_terms: ontology terms that need to be trimmed off

    Output
    -----------------
    term_dict: the trimmed DAG
    """

    # separate dict for terms only
    term_term_dict = {key: DAG[key] for key in list(DAG.keys()) if key in all_terms}

    # separate dict for genes only
    term_gene_dict = {key: DAG[key] for key in list(DAG.keys()) if key not in all_terms}   
    
    # reverse the separate dicts
    term_dict_rev = reverse_graph(term_term_dict)
    gene_dict_rev = reverse_graph(term_gene_dict)

    # run the trim_term function over all terms to update the dicts
    for t in trim_terms:
        print(t)
        trim_term_bottom(t, term_term_dict, term_dict_rev, gene_dict_rev)

    # reverse back the dicts and combine
    term_dict = reverse_graph(term_dict_rev)
    gene_dict = reverse_graph(gene_dict_rev)
    term_dict.update(gene_dict)

    return term_dict


# DAG trimming from top given a list of terms

###--------------> helper function to trim off one term

def trim_term_top(term, term_dict_rev, gene_dict_rev):

    # delete the term
    if term in list(term_dict_rev.keys()):
        del term_dict_rev[term]
    if term in list(gene_dict_rev.keys()):
        del gene_dict_rev[term]


###--------------> function to trim the DAG

def trim_DAG_top(DAG, all_terms, trim_terms):
    """
    Input
    -----------------
    DAG: the DAG to be trimmed
    all_terms: all ontology terms 
    trim_terms: ontology terms that need to be trimmed off

    Output
    -----------------
    term_dict: the trimmed DAG
    """

    # separate dict for ontology terms only
    term_term_dict = {key: DAG[key] for key in list(DAG.keys()) if key in all_terms}

    # separate dict for genes only
    term_gene_dict = {key: DAG[key] for key in list(DAG.keys()) if key not in all_terms}   
    
    # reverse the separate dicts
    term_dict_rev = reverse_graph(term_term_dict)
    gene_dict_rev = reverse_graph(term_gene_dict)

    # run the trim_term function over all terms to update the dicts
    for t in trim_terms:
        print(t)
        trim_term_top(t, term_dict_rev, gene_dict_rev)

    # reverse back the dicts and combine
    term_dict = reverse_graph(term_dict_rev)
    gene_dict = reverse_graph(gene_dict_rev)
    term_dict.update(gene_dict)

    return term_dict




# recursive path-finding function (https://www.python-kurs.eu/hanser-blog/examples/graph2.py)
# in the dict: from key to value, eg. in dag, from child -> parent
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for p in new_paths: 
                paths.append(p)
    return paths




###--------------------------------------
## ACCESS PACKAGE DATA
###--------------------------------------

def data_path():
    path = pkg_resources.resource_filename(__name__, 'data/')
    return path