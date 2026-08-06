import argparse
import configparser
import representation as rep
import heuristics as nelight

from imports import *
from helpers import *


@F.udf(T.ArrayType(T.StringType()))
def get_qids(name):
    return name['ids'][1:-1].split(', ')


IScore = nelight.IntersectionScore()


@F.udf(T.DoubleType())
def iscore_udf(article_representation, entity_representation):
    return len(set(article_representation).intersection(set(entity_representation)))


@F.udf(T.DoubleType())
def eeiscore_udf(article_representation, entity_representation):
    article_representation = map(tuple, article_representation)
    entity_representation = map(tuple, entity_representation)
    return len(set(article_representation).intersection(set(entity_representation)))


@F.udf(T.MapType(T.StringType(), T.ArrayType(T.DoubleType())))
def scores2dict(iscore, niscore, eeiscore, ns, prwp):
    return dict(iscore=iscore, niscore=niscore, eeiscore=eeiscore, ns=ns, prwp=prwp)

@F.udf(T.MapType(T.StringType(), T.ArrayType(T.DoubleType())))
def zero_scores():
    return dict(iscore=0, niscore=0, eeiscore=0, ns=0, prwp=0)

name_struct = T.StructType([
    T.StructField('name', T.StringType(), True),
    T.StructField('ids', T.ArrayType(T.StringType()), True),
    T.StructField('scores', T.MapType(T.StringType(), T.ArrayType(T.DoubleType())), True),
    T.StructField('offsets', T.ArrayType(T.ArrayType(T.ArrayType(T.IntegerType()))), True)])

@F.udf(name_struct)
def get_name_struct(name, ids, scores, offsets):
    return [name, ids, scores, offsets]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.txt')
    spark_config = config['spark']

    n_threads = spark_config['n_threads']
    spark = SparkSession.builder.master(f'local[{n_threads}]').appName('Article representation computation')

    for i, j in config.items():
        if i == 'n_threads':
            continue
        spark = spark.config(i, j)

    spark = spark.getOrCreate()
    rc_config = config['representation_computation']
    article_representations_path = rc_config['article_representations_output']
    qb_articles = rc_config['quotebank_input']
    qb_scores_out = rc_config['scores_out']

    ts_cache_parquet = rc_config['token_set_representation_parquet_path']
    s_cache_parquet = rc_config['statement_representation_parquet_path']
    ns_cache_parquet = rc_config['ns_parquet_path']
    prwp_cache_parquet = rc_config['prwp_parquet_path']

    article_representations = spark.read.parquet(article_representations_path)
    ts_cache = spark.read.parquet(ts_cache_parquet)
    s_cache = spark.read.parquet(s_cache_parquet)
    ns_cache = spark.read.parquet(ns_cache_parquet)
    prwp_cache = spark.read.parquet(prwp_cache_parquet)

    unambiguous_scores = article_representations\
        .withColumn('name', F.explode('names'))\
        .withColumn('qids', get_qids('name'))\
        .where(F.size('qids') > 1)\
        .withColumn('n', F.col('name').getItem('name'))\
        .withColumn('offsets', F.col('name').getItem('offsets'))\
        .withColumn('qid', F.explode('qids'))\
        .join(ts_cache.withColumnRenamed('representation', 'token_set_entity_representation'), on='qid')\
        .withColumn('iscore', iscore_udf('token_set_article_representation', 'token_set_entity_representation'))\
        .withColumn('niscore', iscore_udf('narrow_representation', 'token_set_entity_representation'))\
        .drop('token_set_entity_representation').drop('token_set_article_representation').drop('narrow_representation')\
        .join(s_cache.withColumnRenamed('representation', 'statement_entity_representation'), on='qid')\
        .withColumn('iscore', eeiscore_udf('unambiguous_statements_representation', 'statement_entity_representation'))\
        .drop('unambiguous_statements_representation').drop('statement_entity_representation')\
        .join(ns_cache.withColumnRenamed('representation', 'ns'), on='qid')\
        .join(prwp_cache.withColumnRenamed('representation', 'prwp'), on='qid')

    ambiguous_scores = article_representations\
        .withColumn('name', F.explode('names'))\
        .withColumn('qids', get_qids('name'))\
        .where(F.size('qids') > 1)\
        .withColumn('n', F.col('name').getItem('name'))\
        .withColumn('offsets', F.col('name').getItem('offsets'))\
        .withColumn('qid', F.explode('qids'))\
        .withColumn('scores', zero_scores())\
        .withColumn('iscore', F.col('scores').getItem('iscore'))\
        .withColumn('niscore', F.col('scores').getItem('niscore'))\
        .withColumn('eeiscore', F.col('scores').getItem('eeiscore'))\
        .withColumn('ns', F.col('scores').getItem('ns'))\
        .withColumn('prwp', F.col('scores').getItem('prwp')) \

    scores = ambiguous_scores.union(unambiguous_scores)\
        .groupby('articleID', 'n').agg(F.collect_list('iscore').alias('iscore'),
                                       F.collect_list('niscore').alias('niscore'),
                                       F.collect_list('eeiscore').alias('eeiscore'),
                                       F.collect_list('ns').alias('ns'),
                                       F.collect_list('prwp').alias('prwp'),
                                       F.collect_list('qid').alias('qids'),
                                       F.first('offsets').alias('offsets'))\
        .withColumn('scores', scores2dict('iscore', 'niscore', 'eeiscore', 'ns', 'prwp'))\
        .withColumn('name', get_name_struct('n', 'qids', 'scores', 'offsets'))\
        .groupby('articleID').agg(F.collect_list('name').alias('names'))

    articles = spark.read.parquet(qb_articles).select('articleID', 'quotations')\
        .join(scores, on='articleID').write.parquet(qb_scores_out)