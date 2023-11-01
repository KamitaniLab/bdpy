'''ML models'''


__all__ = []

import copy
from itertools import product

from bdpy.preproc import select_top
import numpy as np
from numpy.linalg import norm
from scipy import stats
from sklearn.svm import SVC
from tqdm import tqdm
import warnings


class EnsembleClassifier(object):
    '''Ensemble classifier.'''

    def __init__(
            self,
            model=SVC(kernel='linear'),
            n_estimators=11,
            n_feat=100,
            normalize_X=False,
            undersampling=True
    ):
        self._classes = {}
        self._model = model
        self._estimators = {}
        self._n_estimators = n_estimators
        self._X_normalization = normalize_X
        self._undersampling = undersampling
        self._n_feat = n_feat
        self._n_targets = 1
        self._target_shape = None
        self._target_order = 'F'
        self._randobj = np.random.RandomState()

    def fit(self, X, Y):
        '''
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        Y : array of shape (n_smaples,) or (n_samples, i, j, k, ...)
          i * j * k * ... = n_targets

        Returns
        -------
        self
        '''

        if Y.ndim == 1:
            self._n_targets = 1
            self._estimators.update({0: {}})
            return self._fit(X, Y)

        self._target_shape = Y.shape[1:]
        Y = Y.reshape(
            Y.shape[0], -1,
            order=self._target_order
        )  # (n_samples, n_targets)
        self._n_targets = Y.shape[1]

        for i in tqdm(range(self._n_targets)):
            self._estimators.update({i: {}})
            self._fit(X, Y[:, i], target=i)

        return self

    def _fit(self, X, y, target=0):
        '''
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)
        target : int
          Zero-based index of a target variable.

        Returns
        -------
        self
        '''

        self._classes.update({target: np.unique(y)})
        y_pairs = self.__get_pairs(self._classes[target])

        for y0, y1 in y_pairs:
            print(f'Target pairs: {y0}, {y1}')

            index = ((y == y0) + (y == y1)).ravel()
            Xs = X[index, :]
            ys = y[index]
            ys[ys == y0] = 0
            ys[ys == y1] = 1

            if np.sum(ys == 0) == np.sum(ys == 1):
                n_est = 1
            else:
                n_est = self._n_estimators

            print(f'Num estimators: {n_est}')

            self._estimators[target].update({(y0, y1): []})

            for n in range(n_est):
                # Undersampling
                if self._undersampling:
                    Xsn, ysn = self.__undersample(Xs, ys)
                else:
                    Xsn, ysn = Xs, ys

                # Normalization
                if self._X_normalization:
                    x_mean = np.mean(Xsn, axis=0)
                    x_std = np.std(Xsn, axis=0, ddof=1)
                    Xsn = (Xsn - x_mean) / x_std
                else:
                    x_mean, x_std = None, None

                # Feature selection
                if self._n_feat == 0 or self._n_feat is None:
                    feature_index = None
                else:
                    feature_index = self.__voxel_selection(
                        Xsn, ysn,
                        n_voxel=self._n_feat
                    )
                    Xsn = Xsn[:, feature_index]

                # Model training
                model = copy.deepcopy(self._model)
                model.fit(Xsn, ysn)

                # Save trained model
                self._estimators[target][(y0, y1)].append({
                    'model': model,
                    'x_mean': x_mean,
                    'x_std': x_std,
                    'selected_features': feature_index,
                })

        return self

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        y_pred : array of shape (n_samples,)
        '''
        if self._n_targets == 1:
            return self._predict(X)

        y_pred = np.vstack([
            self._predict(X, target=i)
            for i in range(self._n_targets)
        ]).T

        y_pred = y_pred.reshape(
            (y_pred.shape[0], ) + self._target_shape,
            order=self._target_order
        )

        return y_pred

    def _predict(self, X, target=0):
        '''
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        target : int
          Zero-based index of a target variable.

        Returns
        -------
        y_pred : array of shape (n_samples,)
        '''

        pred_pairs = []
        dv_pairs = []
        for (y0, y1), estimators in self._estimators[target].items():
            dv_all = []
            for estimator in estimators:
                if self._X_normalization:
                    X_ = (X - estimator['x_mean']) / estimator['x_std']
                else:
                    X_ = X

                if estimator['selected_features'] is not None:
                    X_ = X_[:, estimator['selected_features']]

                dv = estimator['model'].decision_function(X_)  # Expected to be (n_samples,)
                if dv.ndim > 1:
                    warnings.warn(f"The return of decision_function is expected to be one-dimensional with a shape of (n_samples,), but it has two (or more) dimensions. I assume the second column represents the decision function for '{y1}' (i.e., dv = dv[:, 1]). This may be unintended behavior. Please check the returns of the decision_function of your model.")
                    dv = dv[:, 1]
                if isinstance(estimator['model'], SVC):
                    print('OK')
                    dv /= norm(estimator['model'].coef_)
                    # See https://stats.stackexchange.com/questions/14876/interpreting-distance-from-hyperplane-in-svm
                dv_all.append(dv)
            dv_all = np.vstack(dv_all).T  # (n_samples, n_estimators)
            dv_mean = np.mean(dv_all, axis=1, keepdims=True)

            pred = np.zeros(dv_mean.shape)
            pred[dv_mean >= 0] = y1
            pred[dv_mean < 0] = y0

            pred_pairs.append(pred)
            dv_pairs.append(dv_mean)

        pred_pairs = np.hstack(pred_pairs)  # (n_samples, n_pairs)
        dv_pairs = np.hstack(dv_pairs)      # (n_samples, n_pairs)

        # Voting
        y_pred = self.__voting(pred_pairs, dv_pairs, target=target)

        return y_pred

    def __get_pairs(self, classes):
        return [(y0, y1) for y0, y1 in product(classes, classes) if y0 < y1]

    def __undersample(self, x, y):
        '''The original version was implemented by Misato Tanaka.'''

        y_uniq = np.unique(y)
        min_sample_num = np.min([np.sum(y == u) for u in y_uniq])

        new_x = []
        new_y = []
        for u in y_uniq:
            if np.sum(y == u) > min_sample_num:
                tmp_x = x[y == u]
                tmp_y = y[y == u]
                select_ind = np.arange(tmp_x.shape[0])
                self._randobj.shuffle(select_ind)

                new_x.append(tmp_x[select_ind[:min_sample_num]])
                new_y.append(tmp_y[select_ind[:min_sample_num]])
            else:
                new_x.append(x[y == u])
                new_y.append(y[y == u])
        x = np.vstack(new_x)
        y = np.hstack(new_y)

        return x, y

    def __voxel_selection(self, x, y, n_voxel=100):
        '''Voxel selection based on f values of one-way ANOVA.'''
        y_uniq = np.unique(y)
        x_sub = [x[y == k, :] for k in y_uniq]
        f, p = stats.f_oneway(*x_sub)
        _, index = select_top(x, f, n_voxel, axis=1, verbose=False)
        return index

    def __voting(self, y, dv, target=0):
        '''
        Parameters
        ----------
        y : array of (n_samples, n_pairs)
        dv : array of (n_samples, n_pairs)

        Returns
        -------
        y_pred : array of (n_samples)
        '''
        vote = []  # (n_samples, n_classes)
        dv = np.abs(dv)
        for _y, _dv in zip(y, dv):
            vote.append([
                np.sum(_dv[_y == c])
                for c in self._classes[target]
            ])
        vote = np.vstack(vote)  # (n_samples, n_classes)

        y_pred_index = np.argmax(vote, axis=1)  # (n_samples,)
        y_pred = np.zeros(y_pred_index.shape)
        for i, c in enumerate(self._classes[target]):
            y_pred[y_pred_index == i] = c

        return y_pred
