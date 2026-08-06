import representation as rep
import configparser
from helpers import load_pickle


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.txt')
    cache_args = config['representation_computation']
    wikidata_dump_path = cache_args['wikidata_dump_path']
    redirect_tree_path = cache_args['redirect_tree_path']

    wikidata_labels = load_pickle(cache_args['wikidata_labels_path'])
    wikidata_aliases = load_pickle(cache_args['wikidata_aliases_path'])
    wikidata_descriptions = load_pickle(cache_args['wikidata_aliases_path'])

    ts_cache = rep.TokenSetRepresentationCache.from_dump(wikidata_dump_path)
    s_cache = rep.StatementRepresentationCache.from_dump(wikidata_dump_path)
    ns_cache = rep.NSCache.from_dump(wikidata_dump_path)
    prwp_cache = rep.PRCache.from_ranks_file(wikidata_dump_path)

    redirect_resolver = rep.RedirectResolver(redirect_tree_path)

    ts_cache = redirect_resolver.resolve_redirects(ts_cache)
    s_cache = redirect_resolver.resolve_redirects(s_cache)
    ns_cache = redirect_resolver.resolve_redirects(ns_cache)
    prwp_cache = redirect_resolver.resolve_redirects(prwp_cache)

    ts_cache_pkl = cache_args['token_set_representation_pickle_path']
    s_cache_pkl = cache_args['statement_representation_pickle_path']
    ns_cache_pkl = cache_args['ns_pickle_path']
    prwp_cache_pkl = cache_args['prwp_pickle_path']

    ts_cache_parquet = cache_args['token_set_representation_parquet_path']
    s_cache_parquet = cache_args['statement_representation_parquet_path']
    ns_cache_parquet = cache_args['ns_parquet_path']
    prwp_cache_parquet = cache_args['prwp_parquet_path']

    ts_cache.to_pickle(ts_cache_pkl)
    s_cache.to_pickle(s_cache_pkl)
    ns_cache.to_pickle(ns_cache_pkl)
    prwp_cache.to_pickle(prwp_cache_pkl)

    ts_cache.to_parquet(ts_cache_parquet)
    s_cache.to_parquet(s_cache_parquet)
    ns_cache.to_parquet(ns_cache_parquet)
    prwp_cache.to_parquet(prwp_cache_parquet)