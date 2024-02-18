from langchain.schema import format_document
from langchain_community.document_loaders import PySparkDataFrameLoader
from pyspark.sql import functions as F

from utils.templates import DEFAULT_DOCUMENT_PROMPT

def get_data(path,spark):
    df = spark.read.json(path)
    df = df.drop('hours').drop('business_id').drop("attributes")
    df = df.withColumn('fulladdress',F.concat(F.col('address'),F.lit(', '),F.col('city'),F.lit(', '),F.col('state'),F.lit(', '),F.col('postal_code')))
    df = df.drop('latitude','longitude','postal_code','address','city','state','is_open','review_count')
    df = df.dropna()
    df = df.limit(20)
    loader = PySparkDataFrameLoader(spark,df,page_content_column='name')
    docs = loader.load()
    return docs

def combine_docs(docs,default_formatter=DEFAULT_DOCUMENT_PROMPT):
    docs_string = [format_document(doc,default_formatter) for doc in docs]
    return '\n\n'.join(docs_string)
