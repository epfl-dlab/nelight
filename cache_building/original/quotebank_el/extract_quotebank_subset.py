import gzip
import json
from tqdm import tqdm
import configparser
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T


@F.udf(T.ArrayType(T.StringType()))
def get_qids(name):
    return name['ids'][1:-1].split(', ')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.txt')
    cache_args = config['representation_computation']
    articles_path = cache_args['quotebank_input']
    wikidata_dump_path = cache_args['wikidata_dump_path']
    wikidata_raw_dump_path = cache_args['wikidata_raw_dump_path']

    config = configparser.ConfigParser()
    config.read('config.txt')
    spark_config = config['spark']

    n_threads = spark_config['n_threads']
    spark = SparkSession.builder.master(f'local[{n_threads}]').appName('Article representation computation')

    for i, j in config.items():
        if i == 'n_threads':
            continue
        spark = spark.config(i, j)

    articles = spark.read.parquet(articles_path)

    qids = set(map(lambda x: x.qid, articles.select(F.explode('names').alias('name'))\
                   .select(get_qids('name').alias('qid'))\
                   .select(F.explode('qid').alias('qid'))\
                   .collect()))

    cache = {}
    total_qids = len(qids)

    with gzip.open(wikidata_raw_dump_path, 'rb') as f:
        lines = []
        with gzip.open(wikidata_dump_path, 'wb') as f_out:
            pbar = tqdm(enumerate(f))
            for i, instance in pbar:
                pbar.set_postfix({'QIDs processed': f'{(1 - len(qids) / total_qids) * 100:.2f}%'})
                instance_decoded = instance.decode('utf-8')
                instance_decoded = instance_decoded[:-2]
                if len(instance_decoded) == 0:
                    continue
                try:
                    instance_decoded = json.loads(instance_decoded)
                except json.decoder.JSONDecodeError:
                    print(instance_decoded)
                    continue

                qid = instance_decoded['id']
                if qid not in qids:
                    continue
                qids.remove(qid)
                lines.append(instance)

                if len(qids) == 0:
                    break
        f_out.writelines(lines)
