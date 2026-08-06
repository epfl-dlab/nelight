from opentapioca.wikidatagraph import WikidataGraph
import numpy as np

def calculate_pagerank():
    graph = WikidataGraph()
    print('Loading from preprocessed dump...')
    graph.load_from_preprocessed_dump('/dlabdata1/culjak/wikidata_graph.tsv')
    print('Computing pagerank...')
    graph.compute_pagerank()
    print('Saving pagerank...')
    graph.save_pagerank('/dlabdata1/culjak/pagerank.npy')
    print('Done!')
calculate_pagerank()
# graph = WikidataGraph()
# graph.load_from_preprocessed_dump('/dlabdata1/culjak/wikidata_graph.tsv')
# graph.load_pagerank('/dlabdata1/culjak/pagerank.npy')
# print(graph.get_pagerank('Q4914'))