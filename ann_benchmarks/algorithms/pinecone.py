import numpy as np
import random
import pinecone

from .base import BaseANN

INDEX_CHUNK_SIZE = 10000


def metric_mapping(_metric: str):
    _metric_type = {"angular": "cosine", "euclidean": "euclidean"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


class Pinecone(BaseANN):
    def __init__(self, metric, dim, index_param):
        pinecone.init(api_key="473f4e1a-1388-4a2c-abd3-e08c22a80fa1", environment="us-east-1-aws")
        self._index_name = f"ann_benchmark_{metric}_{dim}_{random.randint(0, 1000000)}"
        self._dim = dim
        self._metric_type = metric_mapping(metric)


    def fit(self, X):
        self.freeIndex()
        pinecone.create_index(
            self._index_name, 
            dimension=self._dim, 
            metric=self._metric_type,
            pods=2,
            replicas=1,
            pod_type="p2.x1"
        )
        index = pinecone.Index(self._index_name)
        dataset_size = len(X)
        print(f"dataset size: {dataset_size}")
        i = 0
        while i < dataset_size:
            ceiling = min(i+INDEX_CHUNK_SIZE, dataset_size)
            print(f"adding data from row {i} to {ceiling}")
            index.upsert([
                (i, X[i])
                for i in range(i, ceiling)
            ])
            i += INDEX_CHUNK_SIZE


    def query(self, v, n):
        index = pinecone.Index(self._index_name)
        query_response = index.query(
            vector=v, 
            top_k=n,
            include_values=False,
            include_metadata=False,
        )
        return [m.id for m in query_response.matches]


    def freeIndex(self):
        try:
            pinecone.delete_index(self._index_name)
        except Exception as e:
            print("no index to delete")


    def done(self):
        pass
