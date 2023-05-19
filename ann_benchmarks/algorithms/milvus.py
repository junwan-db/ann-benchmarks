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

INDEX_CHUNK_SIZE = 250000

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
        self._collection_name = "ann_benchmarks_test_milvus"
        connections.connect("default", host="a5e9f321a3b4249cc857193c97a28820-1115516199.us-west-2.elb.amazonaws.com", port="19530")

        fields = [
          FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
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
        while i < dataset_size:
          print(f"ingesting data rows from {i} to {min(i+INDEX_CHUNK_SIZE, dataset_size)}")
          self._milvus_collection.insert([X[i:min(i+INDEX_CHUNK_SIZE, dataset_size)]])
          i += INDEX_CHUNK_SIZE

        self._milvus_collection.flush()

        print("added to collection, creating index")

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
        index_created = False
        while not index_created:
          progress = utility.index_building_progress(self._collection_name)
          if progress["indexed_rows"] < progress["total_rows"]:
            print(f"waiting for index to build, indexed rows: {progress['indexed_rows']}, total rows: {progress['total_rows']}")
            time.sleep(5)
          else:
            print("indexing complete")
            index_created = True

        print("created index, loading to memory")
        self._milvus_collection.load()

        # wait for index to be loaded to memory
        index_loaded_in_memory = False
        while not index_loaded_in_memory:
          progress = utility.load_state(self._collection_name)
          if progress["loading_progress"] != "100%":
            print(f"waiting for index to load in memory, progress: {progress['loading_progress']}")
            time.sleep(5)
          else:
            print("index loaded complete")
            index_loaded_in_memory = True


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
