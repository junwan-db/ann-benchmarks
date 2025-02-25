from time import sleep
from urllib.request import Request, urlopen

from opensearchpy import ConnectionError, OpenSearch
from opensearchpy.helpers import bulk
from opensearchpy.connection import Urllib3HttpConnection
from tqdm import tqdm
import requests
from requests.auth import HTTPBasicAuth
from .base import BaseANN


class OpenSearchKNN(BaseANN):
    def __init__(self, metric, dimension, method_param):
        self.metric = {"angular": "cosinesimil", "euclidean": "l2"}[metric]
        self.dimension = dimension
        self.method_param = method_param
        self.param_string = "-".join(k + "-" + str(v) for k, v in self.method_param.items()).lower()
        self.name = f"os-{self.param_string}"
        self.host = 'http://ad927cdbc47e14edea28b2f800c28348-1231483935.us-west-2.elb.amazonaws.com'
        self.client = OpenSearch(
          hosts = [self.host],
          http_auth = ('admin', 'admin'),
          use_ssl = False,
          retry_on_timeout=True,
          timeout=100
        )
        self._wait_for_health_status()

    def _wait_for_health_status(self, wait_seconds=1500, status="yellow"):
        for _ in range(wait_seconds):
            try:
                self.client.cluster.health(wait_for_status=status)
                return
            except ConnectionError as e:
                pass
            sleep(1)

        raise RuntimeError("Failed to connect to OpenSearch")

    def fit(self, X):
        body = {
            "settings": {"index": {"knn": True}, "number_of_shards": 3, "number_of_replicas": 0}
        }

        mapping = {
            "properties": {
                "id": {"type": "keyword", "store": True},
                "vec": {
                    "type": "knn_vector",
                    "dimension": self.dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": self.metric,
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": self.method_param["efConstruction"],
                            "m": self.method_param["M"],
                        },
                    },
                },
            }
        }

        

        self.freeIndex()

        print("Creating new index Index:", self.name)
        self.client.indices.create(self.name, body=body, request_timeout=1000)
        self.client.indices.put_mapping(mapping, self.name, request_timeout=1000)

        self._wait_for_health_status()
        print("Uploading data to the Index:", self.name)

        def gen():
            for i, vec in enumerate(tqdm(X)):
                yield {"_op_type": "index", "_index": self.name, "vec": vec.tolist(), "id": str(i + 1)}

        (_, errors) = bulk(self.client, gen(), chunk_size=3000, max_retries=10, request_timeout=1000, refresh="wait_for")
        assert len(errors) == 0, errors

        print("Force Merge...")
        self.client.indices.forcemerge(self.name, max_num_segments=5, request_timeout=1000)

        print("Dummy searching to ensure refresh: ", self.name)
        dummy_search_response = self.client.search(index=self.name)
        
        print("Turning off refreshing for: ", self.name)
        self.client.indices.put_settings(index=self.name, body={
                "index.refresh_interval": "-1"
            })

        print("Running Warmup API...")

        response = requests.get(self.host + ":9200/_plugins/_knn/warmup/" + self.name + "?pretty", 
                                verify=False, 
                                auth=HTTPBasicAuth('admin', 'admin'))
        print(response.text)
        self._wait_for_health_status()

    def set_query_arguments(self, ef):
        body = {"settings": {"index": {"knn.algo_param.ef_search": ef}}}
        self.client.indices.put_settings(body=body, index=self.name)

    def query(self, q, n):
        body = {"query": {"knn": {"vec": {"vector": q.tolist(), "k": n}}}}

        res = self.client.search(
            index=self.name,
            body=body,
            size=n,
            _source=False,
            docvalue_fields=["id"],
            stored_fields="_none_",
            filter_path=["hits.hits.fields.id"],
            request_timeout=10,
        )
        # print(f"=finished one query :{res}")
        return [int(h["fields"]["id"][0]) - 1 for h in res["hits"]["hits"]]

    def batch_query(self, X, n):
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

    def freeIndex(self):
        if (self.client.indices.exists(index=self.name)):
            print(f"Removing old index: {self.name}")
            self.client.indices.delete(index=self.name, request_timeout=1000)
