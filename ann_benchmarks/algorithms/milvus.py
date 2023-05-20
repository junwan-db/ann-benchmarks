import numpy as np
import pymilvus
import time
import uuid
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

# import pyknowhere

from .base import BaseANN

INDEX_CHUNK_SIZE = 10000

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
        connections.connect("default", host="a21d3a5d0c499434f9310b66869abb07-675851668.us-west-2.elb.amazonaws.com", port="19530")

        fields = [
          FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
          FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self._dim)
        ]
        schema = CollectionSchema(fields, "benchmarking")
        self._milvus_collection = Collection(self._collection_name, schema)
        print("created collection, releasing index from previous run")
        # release index from previous run
        try:
          self._milvus_collection.release()
          self._milvus_collection.drop_index()
        except Exception as e:
          print(f"failed to release previous index with error {str(e)}")
        print("init complete")


    def fit(self, X):
        # self.client = pyknowhere.Index(self._metric_type, self._dim, len(X), self._index_m, self._index_ef)
        # self.client.add(X, numpy.arange(len(X)))

        dataset_size = len(X)
        print(f"dataset size: {dataset_size}")

        print("adding data to collection")
        i = 0
        while i < dataset_size:
          ceiling = min(i+INDEX_CHUNK_SIZE, dataset_size)
          print(f"adding data from row {i} to {ceiling}")
          chunk = X[i:ceiling]
          self._milvus_collection.insert([
            list(range(i, ceiling)),
            chunk
          ])
          i += INDEX_CHUNK_SIZE

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
            time.sleep(3)
          else:
            print("indexing complete")
            index_created = True

        print("created index, loading to memory")
        self._milvus_collection.load()

        # wait for index to be loaded to memory
        index_loaded_in_memory = False
        while not index_loaded_in_memory:
          progress = utility.loading_progress(self._collection_name)
          if progress["loading_progress"] != "100%":
            print(f"waiting for index to load in memory, progress: {progress['loading_progress']}")
            time.sleep(3)
          else:
            print("index loaded complete")
            index_loaded_in_memory = True


    def set_query_arguments(self, ef):
        self._search_ef = ef
        # self.client.set_param(ef)


    def query(self, v, n):
        # return self.client.search(v, k=n)
        res = self.batch_query([v], n)[0]
        # print(f"query result: {res}")
        return res
    

    def batch_query(self, X, n):
        # return self.client.search(v, k=n)
        results = self._milvus_collection.search(
          data=X,
          anns_field="embeddings",
          param={
            "metric_type": "IP",
            "params": {
              "ef": self._search_ef
            }
          },
          limit=n,
          expr=None
        )
        self.batch_results = [
          [
            int(hit.id)
            for hit in result
          ]
          for result in results
        ]
        return self.batch_results
    
    def get_batch_results(self):
        return self.batch_results

    def __str__(self):
        return f"Milvus(index_M:{self._index_m},index_ef:{self._index_ef},search_ef={self._search_ef})"
