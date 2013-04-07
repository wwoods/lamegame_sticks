
import logging
import numpy as np
import sklearn.utils
from sklearn.base import BaseEstimator, RegressorMixin

import nn
from voters import EcoSystemVar

class nn_model(BaseEstimator, RegressorMixin):
    def __init__(self, alpha = 0.01, max_iter = 1):
        self._voters = EcoSystemVar(nn.NeuralNet)
        self._lastOutputLen = 0
        self._alpha = alpha
        self._max_iter = max_iter
        self._met_alpha_iterations = None
        self._trainSets = {}


    def fit(self, X, y):
        """Fit model."""
        X, y = sklearn.utils.check_arrays(X, y, sparse_format = "dense",
                dtype = np.float)

        insAndOuts = zip(X, y)
        for i in range(self._max_iter):
            for n, out in insAndOuts:
                self._trainVoters(n, out)
                self._lastOutputLen = len(out)

            # Get max distance to points
            maxError = 0
            for n, out in insAndOuts:
                r = self._voters.eval(n, len(out))
                rerr = sum([ (rv - ov) ** 2 for rv, ov in zip(r, out) ]) ** 0.5
                maxError = max(rerr, maxError)
            if maxError < self._alpha:
                self._met_alpha_iterations = i
                break


    def met_alpha(self):
        """Returns None if alpha was not met, or the number of iterations
        required.
        """
        return self._met_alpha_iterations


    def predict(self, X):
        """Returns y for each X."""
        X = sklearn.utils.array2d(X)
        return [ self._voters.eval(i, self._lastOutputLen)
                for i in X ]


    def _trainVoters(self, inputs, outputs):
        trainSets = self._trainSets

        worst = trainSets.setdefault('worst', [])
        worst.append(TrainRecord(inputs, outputs))
        for w in worst:
            w.update(self._voters)
            # Worst get "less" bad as they age
            a = 400.0
            w.error *= a / (a + w.age)
        worst.sort(key = lambda m: -m.error)
        trainSets['worst'] = worst[:5]

        best = trainSets.setdefault('best', [])
        best.append(TrainRecord(inputs, outputs))
        for b in best:
            b.update(self._voters)
            # Pride
            a = 400.0
            b.error /= a / (a + b.age)
        best.sort(key = lambda m: m.error)
        trainSets['best'] = best[:5]

        # A training record is "diverse" if its output differences over its input
        # differences is substantial.
        diverse = trainSets.setdefault('diverse', [])
        diverse.append(TrainRecord(inputs, outputs))
        for d in diverse:
            totalOutput = 0.0
            totalInput = 0.0
            for d2 in diverse:
                for a, b in zip(d.outputs, d2.outputs):
                    totalOutput += (a - b) ** 2
                for a, b in zip(d.inputs, d2.inputs):
                    totalInput += (a - b) ** 2
            d.diversity = totalOutput / max(1e-30, totalInput)
            d.update(self._voters)
        diverse.sort(key = lambda m: -m.diversity)
        trainSets['diverse'] = diverse[:5]

        recent = trainSets.setdefault('recent', [])
        recent.append(TrainRecord(inputs, outputs))
        trainSets['recent'] = recent[-3:]
        for r in trainSets['recent']:
            r.update(self._voters)

        before = 0
        beforeRecent = 0
        for s in trainSets:
            for e in trainSets[s]:
                before += e.error
                if s == 'recent':
                    beforeRecent += e.error

        # For each iteration: worst x 4, best x 2, recent x 1
        for _ in range(1):
            for _ in range(2):
                for w in trainSets['worst']:
                    self._voters.train(w.inputs, w.outputs)
            for _ in range(1):
                for b in trainSets['best']:
                    self._voters.train(b.inputs, b.outputs)
            for _ in range(1):
                for b in trainSets['diverse']:
                    self._voters.train(b.inputs, b.outputs)
            for _ in range(1):
                for r in trainSets['recent']:
                    self._voters.train(r.inputs, r.outputs)

        after = 0
        afterRecent = 0
        sets = 0
        for s in trainSets:
            for e in trainSets[s]:
                e.update(self._voters, errorOnly = True)
                if s == 'worst':
                    # Worst get "less" bad as they age
                    a = 400.0
                    e.error *= a / (a + e.age)
                elif s == 'best':
                    a = 400.0
                    e.error /= a / (a + e.age)
                after += e.error
                if s == 'recent':
                    afterRecent += e.error
                sets += 1

        # Take average sqr error
        before /= sets
        after /= sets

        logging.debug(
                "B/A: {0:.4f}/{1:.4f} - {3:.4f}/{4:.4f} (over {2})".format(
                before, after, sets, beforeRecent, afterRecent))



class TrainRecord(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.error = None
        self.age = 0


    def update(self, net, errorOnly = False):
        r = net.eval(self.inputs, len(self.outputs))
        self.error = 0.0
        for real, expected in zip(r, self.outputs):
            self.error += (real - expected) ** 2

        if not errorOnly:
            self.age += 1
