import argparse
import configparser
import pyspark.sql.functions as F
import pyspark.sql.types as T
import representation as rep

from pyspark.sql import SparkSession
from nltk import word_tokenize


@F.udf(T.ArrayType(T.StringType()))
def get_unambiguous_qids(names):
    qids = []
    for name in names:
        name['ids'] = name['ids'][1:-1].split(', ')
        if len(name['ids']) == 1:
            qids.append(name['ids'][0])
    if len(qids) == 0:
        return [None]
    return qids


unamb_statements_rep_type = T.StructType([
    T.StructField('text', T.StringType(), True),
    T.StructField('offsets', T.ArrayType(T.ArrayType(T.ArrayType(T.IntegerType()))), True)])


@F.udf(unamb_statements_rep_type)
def get_text_offsets(content, names):
    return [content,
            [[list(map(int, i.split(', '))) for i in name['offsets'][2:-2].replace('], [', '|').split('|')] for name in
             names]]


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
    in_path = rc_config['quotebank_input']
    out_path = rc_config['article_representations_output']

    articles = spark.read.parquet(in_path)
    statement_entity_representations = spark.read.parquet(rc_config['statement_representation_parquet_path'])

    token_set_representation_pipeline = rep.Pipeline().add(lambda x: x.lower()).add(word_tokenize).add(set).add(list)
    narrow_representation_pipeline = rep.Pipeline().add(rep.NarrowTokenizeProcessor()).add(
        token_set_representation_pipeline.branch())
    unambiguous_entity_statements_representation_pipeline = rep.Pipeline().add(rep.UnambiguousEntitiesProcessor())

    token_set_representation_pipeline_udf = F.udf(token_set_representation_pipeline, T.ArrayType(T.StringType()))
    narrow_representation_pipeline_udf = F.udf(narrow_representation_pipeline, T.ArrayType(T.ArrayType(T.StringType())))
    unambiguous_entity_statements_representation_pipeline_udf = F.udf(
        unambiguous_entity_statements_representation_pipeline, T.ArrayType(T.ArrayType(T.StringType())))

    articles.withColumn('unambiguous_qids', get_unambiguous_qids('names')) \
        .withColumn('qid', F.explode('unambiguous_qids')) \
        .join(statement_entity_representations, on='qid', how='left') \
        .withColumn('representation', F.explode('representation')) \
        .groupby('articleID').agg(F.first('articleID').alias('articleID'),
                                  F.first('content').alias('content'),
                                  F.first('names').alias('names'),
                                  F.collect_set('representation').alias('unambiguous_statements_representation')) \
        .withColumn('text_offsets', get_text_offsets('content', 'names')) \
        .withColumn('token_set_representation', token_set_representation_pipeline_udf('content')) \
        .withColumn('narrow_representation', narrow_representation_pipeline_udf('text_offsets')) \
        .select('articleID', 'names', 'token_set_representation', 'narrow_representation', 'unambiguous_statements_representation')\
        .write.parquet(out_path)
