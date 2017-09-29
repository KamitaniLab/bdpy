'''learning module'''


from abc import ABCMeta, abstractmethod

import numpy as np
import copy


#-----------------------------------------------------------------------
class BaseLearning(object):
    '''Base class for learning'''

    __metaclass__ = ABCMeta

    def __init__(self):
        self._preprocessing = []
        self._postprocessing = []

    @abstractmethod
    def run(self, *args, **kargs):
        pass
        
    def add_preprocessing(self, func, args=None):
        '''Add preprocessing function'''
        self._preprocessing.append({'func' : func,
                                    'args' : args})
            

    def add_postprocessing(self, func, args=None):
        '''Add postprocessing function'''
        self._postprocessing.append({'func' : func,
                                     'args' : args})
    

#-----------------------------------------------------------------------
class Classification(BaseLearning):
    '''Classification class
    Parameters
    ----------
    x_train, y_train : array_like
        Training data (features) and target labels
    x_test, y_test : array_like
        Test data (features) and target labels
    classifier
        Classifier
    verbose : {'off', 'info'}, optional
        Verbosity level

    Attributes
    ----------
    classifier_trained
        Trained classifier
    prediction
        Predicted labels
    prediction_accuracy
        Prediction accuracy
   '''


    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 classifier=None, verbose='off'):
        BaseLearning.__init__(self)

        # Parameters
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = classifier
        self.verbose = verbose

        # Results
        self.classifier_trained = None
        self.prediction = None
        self.prediction_accuracy = None
        

    def run(self):
        '''Run classification'''

        self.classifier_trained = copy.deepcopy(self.classifier)
        
        for p in self._preprocessing:
            func = p['func']
            args = p['args']

            if args == None:
                self.x_train, self.y_train, self.x_test, self.y_test \
                    = func(self.x_train, self.y_train, self.x_test, self.y_test)
            else:
                self.x_train, self.y_train, self.x_test, self.y_test \
                    = func(self.x_train, self.y_train, self.x_test, self.y_test, *args)

        self.classifier_trained.fit(self.x_train, self.y_train)
        self.prediction = self.classifier_trained.predict(self.x_test)

        self.prediction_accuracy = self.__calc_accuracy(self.prediction, self.y_test)


    def __calc_accuracy(self, ypred, ytest):
        return float(np.sum(ytest == ypred)) / len(ytest)


#-----------------------------------------------------------------------
class CrossValidation(BaseLearning):
    '''Cross-validation class

    Parameters
    ----------
    x, y : array_like
        Data (features) and target labels
    classifier :
        Classifier
    index : k-folds iterator
        Index iterator for cross-validation
    keep_classifiers : bool, optional
        If True, keep trained classifiers in each fold (default: False)
    verbose : {'off', 'info'}, optional
        Verbosity level (default: 'off')

    Attributes
    ----------
    classifier_trained : list
        Trained classifier in each fold
    prediction_accuracy : list
        Prediction accuracy in each fold
    '''

    def __init__(self, x, y, classifier=None, index=None,
                 keep_classifiers=False, verbose='off'):
        BaseLearning.__init__(self)

        # Parameters
        self.x = x
        self.y = y
        self.classifier = classifier
        self.index = index
        self.keep_classifiers = keep_classifiers
        self.verbose = verbose

        # Results
        self.classifier_trained = []
        self.prediction_accuracy = []
        

    def run(self):
        '''Run cross-validation

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        cls = Classification(x_train=None, y_train=None, x_test=None, y_test=None,
                             classifier=self.classifier, verbose='off')
        for p in self._preprocessing:
            func = p['func']
            args = p['args']

            if args == None:
                cls.add_preprocessing(func)
            else:
                cls.add_preprocessing(func, args=args)
        
        for train_index, test_index in self.index:
            cls.x_train = self.x[train_index, :]
            cls.y_train = self.y[train_index, :].flatten()
            cls.x_test  = self.x[test_index, :]
            cls.y_test  = self.y[test_index, :].flatten()

            cls.run()

            if self.keep_classifiers:
                self.classifier_trained.append(cls.classifier_trained)

            self.prediction_accuracy.append(cls.prediction_accuracy)

        if self.verbose is 'info':
            print('Prediction accuracy: %f' % np.mean(self.prediction_accuracy))
