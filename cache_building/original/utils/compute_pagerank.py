from opentapioca.wikidatagraph import WikidataGraph

graph = WikidataGraph()
print('Loading from a dump...')
graph.load_from_preprocessed_dump('/dlabdata1/culjak/wikidata_graph.tsv')
print('Computing pagerank')
graph.compute_pagerank()
graph.save_pagerank('/dlabdata1/culjak/pagerank2.npy')