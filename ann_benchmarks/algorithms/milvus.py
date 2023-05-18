import numpy as np
import pymilvus
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

# import pyknowhere

from .base import BaseANN

INDEX_CHUNK_SIZE = 500000

def metric_mapping(_metric: str):
    _metric_type = {"angular": "IP", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Milvus(BaseANN):
    """
    Needs `__AVX512F__` flag to run, otherwise the results are incorrect.
    Support HNSW index type
    """

    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)
        self._search_ef = None
        # self.client = None
        self._collection_name = "ann_benchmarks_test"
        connections.connect("default", host="a1554b19dd6be4ce696ca34b5824d2fd-824950804.us-west-2.elb.amazonaws.com", port="19530")
        fields = [
          FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
          FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self._dim)
        ]
        schema = CollectionSchema(fields, "benchmarking")
        self._milvus_collection = Collection(self._collection_name, schema)

    def fit(self, X):
        # self.client = pyknowhere.Index(self._metric_type, self._dim, len(X), self._index_m, self._index_ef)
        # self.client.add(X, numpy.arange(len(X)))

        # build collection
        dataset_size = len(X)
        print(f"dataset size: {dataset_size}")
        i = 0
        while i + INDEX_CHUNK_SIZE <= dataset_size:
          print(f"ingesting data rows from {i} to {min(i+INDEX_CHUNK_SIZE, dataset_size)}")
          self._milvus_collection.insert([X[i:min(i+INDEX_CHUNK_SIZE, dataset_size)]])
          self._milvus_collection.flush()
          i += INDEX_CHUNK_SIZE

        # build index
        index = {
          "index_type": "HNSW",
          "metric_type": self._metric_type,
          "params": {
            "M": self._index_m,
            "efConstruction": self._index_ef
          }
        }
        self._milvus_collection.create_index("embeddings", index)

        # wait for index to be loaded to memory
        index_loaded_in_memory = False
        while not index_loaded_in_memory:
          try:
            self.query(X[0], 1)
            index_loaded_in_memory = True
          except Exception as e:
            if not "has not been loaded to memory or load failed" in str(e):
              raise e
            else:
              print("waiting for index to build")
              time.sleep(10)


    def set_query_arguments(self, ef):
        self._search_ef = ef
        # self.client.set_param(ef)


    def query(self, v, n):
        # return self.client.search(v, k=n)
        res = self.batch_query([v], n)[0]
        print(f"query result: {res}")
        return self.batch_query([v], n)[0]
    
    def batch_query(self, X, n):
        # return self.client.search(v, k=n)
        results = self._milvus_collection.search(
          data=X,
          anns_field="embeddings",
          param={
            "ef": self._search_ef
          },
          limit=n,
          expr=None,
          output_fields=["embeddings"]
        )
        self.batch_results = [
          [
            [
              hit.entity.get("embeddings")
              for hit in hits
            ]
            for hits in result
          ]
          for result in results
        ]
        return self.batch_results
    
    def get_batch_results(self):
        return self.batch_results

    def __str__(self):
        return f"Milvus(index_M:{self._index_m},index_ef:{self._index_ef},search_ef={self._search_ef})"
