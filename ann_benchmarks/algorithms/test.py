import numpy

# import pyknowhere

from .base import BaseANN


class Test(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim

    def fit(self, X):
        print(X)
        print(type(X))

    def set_query_arguments(self, ef):
        self._search_ef = ef

    def query(self, v, n):
        print(v)
        print(type(v))
    
    def batch_query(self, X, n):
        print(X)
    
    def get_batch_results(self):
        return self.batch_results

    def __str__(self):
        return f"Test(index_M:{self._index_m},index_ef:{self._index_ef},search_ef={self._search_ef})"