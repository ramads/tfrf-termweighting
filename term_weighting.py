__author__ = 'rama'
__date__ = '9/21/2015'

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import normalize
from sklearn.feature_extraction.text import CountVectorizer

class BaseTranformer(BaseEstimator, TransformerMixin):
    # default class name
    _pos_class = "positive"
    _neg_class = "negative"

    def __init__(self, norm='l2', use_global_weight = True , sublinear_tf=False):
        self.norm = norm
        self.use_global_weight = use_global_weight
        self.sublinear_tf = sublinear_tf

    def set_positive_class(self, class_name):
        self._pos_class = class_name

    def get_positive_class(self):
        return self._pos_class

    def set_negative_class(self, class_name):
        self._neg_class = class_name

    def get_negative_class(self):
        return self._neg_class

    def transformer_settting(self, norm='l2', use_global_weight = True , sublinear_tf=False):
        self.norm = norm
        self.use_global_weight = use_global_weight
        self.sublinear_tf = sublinear_tf

    def split_to_each_category(self, X, y):
        mat = X.toarray()
        mat1, mat2 = [], []
        for i in range(0, len(y)):
            if y[i] == self._pos_class:
                mat1.append(mat[i])
            elif y[i] == self._neg_class:
                mat2.append(mat[i])
        return csc_matrix(np.array(mat1)), csc_matrix(np.array(mat2))

    # this method is an abstract method and must be overridden in child class.
    # then, define your own '_global_diag' if you want to use global weighting when transform stage
    # _global_diag is a list of global weight for your features
    def fit(self, X, y=None, y_sub = None):
        raise NotImplemented("Should have implemented this");

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_global_weight:
            check_is_fitted(self, '_global_diag', 'global term vector is not fitted')
            expected_n_features = self._global_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._global_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    @property
    def global_weight_(self):
        if hasattr(self, "_global_diag"):
            return np.ravel(self._global_diag.sum(axis=0))
        else:
            return None

class BaseVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm = 'l2', use_global_weight=True, sublinear_tf=False):

        super(BaseVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

    # Broadcast the global weight parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._baseTransformer.norm

    @norm.setter
    def norm(self, value):
        self._baseTransformer.norm = value

    @property
    def use_global_weight(self):
        return self._baseTransformer.use_global_weight

    @use_global_weight.setter
    def use_global_weight(self, value):
        self._baseTransformer.use_rf = value

    @property
    def sublinear_tf(self):
        return self._baseTransformer.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._baseTransformer.sublinear_tf = value

    @property
    def global_weight_(self):
        return self._baseTransformer.global_weight_

    def raw_fit_transform(self, raw_documents):
        return super(BaseVectorizer, self).fit_transform(raw_documents)

    def raw_transform(self, raw_documents):
        return super(BaseVectorizer, self).transform(raw_documents)

    def fit(self, raw_documents, y=None, y_sub = None):
        X = super(BaseVectorizer, self).fit_transform(raw_documents)
        self._baseTransformer.fit(X, y, y_sub)
        return self

    def fit_transform(self, raw_documents, y=None, y_sub = None):
        X = super(BaseVectorizer, self).fit_transform(raw_documents)
        self._baseTransformer.fit(X, y, y_sub)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._baseTransformer.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        check_is_fitted(self, '_baseTransformer', 'The baseTransformer vector is not fitted')
        X = super(BaseVectorizer, self).transform(raw_documents)
        return self._baseTransformer.transform(X, copy=False)

# =======================================================================================================================
# term weighting with TFRF method for 2 class only
class TfRf_Transformer(BaseTranformer):
    def __init__(self, norm='l2', use_global_weight = True , sublinear_tf=False):
        self.transformer_settting(norm=norm, use_global_weight=use_global_weight, sublinear_tf=sublinear_tf)

    def fit(self, X, y = None, y_sub = None):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        n_samples, n_features = X.shape
        X1, X2 = self.split_to_each_category(X, y)

        a = np.bincount(X1.nonzero()[1], minlength=X1.shape[1])
        c = np.bincount(X2.nonzero()[1], minlength=X2.shape[1])
        # replace 0 value in c to 1 == max(1,c)
        c[c == 0] = 1.0

        rf_ = a.astype(float)/c
        rf = np.log2(2 + rf_)

        # input type for spdiag => numpy.array => rf
        self._global_diag = sp.spdiags(rf, diags=0, m=n_features, n=n_features)
        return self

class TfRf_Vectorizer(BaseVectorizer):
    def __init__(self, input='content', encoding='utf-8',
         decode_error='strict', strip_accents=None, lowercase=True,
         preprocessor=None, tokenizer=None, analyzer='word',
         stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
         ngram_range=(1, 1), max_df=1.0, min_df=1,
         max_features=None, vocabulary=None, binary=False,
         dtype=np.int64, norm='l2', use_global_weight=True,
         sublinear_tf=False):

        super(TfRf_Vectorizer, self).__init__(input=input, encoding=encoding,
             decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
             preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
             stop_words=stop_words, token_pattern=token_pattern,
             ngram_range=(1, 1), max_df=1.0, min_df=1,
             max_features=None, vocabulary=vocabulary, binary=binary,
             dtype=np.int64, norm=norm, use_global_weight=True,
             sublinear_tf=sublinear_tf)

        self._baseTransformer = TfRf_Transformer(norm=norm, use_global_weight=use_global_weight, sublinear_tf=sublinear_tf)