
import copy
import math
import random

class EcoMember(object):

    @property
    def error(self):
        return self._error


    def __init__(self, computer):
        self._computer = computer
        self._error = 0.0
        self._confidence = 0.0


    def copy(self):
        return copy.deepcopy(self)


    def eval(self, *args, **kwargs):
        return self._computer.eval(*args, **kwargs)


    def mimic(self, other):
        """Mimic another EcoMember's computation.
        """
        self._computer = other._computer.copy()
        self._error = other._error
        self._confidence = other._confidence


    def mutate(self, rate):
        self._computer.mutate(rate=rate)


    def setError(self, newError):
        self._error = newError


    def train(self, inputs, outputs, rate):
        """Returns an error measure for the given input / output; that error
        does not affect self._error.  That is the responsibility of the caller.
        """
        return self._computer.train(inputs, outputs, rate = rate)


class EcoSystem(object):
    def __init__(self, logicClass, population):
        self._eliteCount = 1
        self._mutateCount = 0
        self._logicClass = logicClass
        self._members = []
        for i in range(population):
            self.addSomeone(0.0)


    def addSomeone(self, rate):
        """Adds a new member to the end of the queue with an error equal to
        twice the last member's error.
        """
        if rate > 0:
            member = self._getRouletteMember().copy()
            member.mutate(rate)
            member.mutate(rate)
            member.mutate(rate)
            member.mutate(rate)
            member.mutate(rate)
        else:
            member = EcoMember(self._logicClass())
        self._members.append(member)


    def eval(self, inputs, numOutputs):
        return self._members[0].eval(inputs, numOutputs)


    def killSomeone(self):
        """Kills the worst member."""
        self._members.pop()


    def train(self, inputs, outputs, rate=0.05):
        for i, m in enumerate(self._members):
            mrate = rate
            if i < self._eliteCount:
                mrate = self._getEliteRate(rate)
            error = m.train(inputs, outputs, rate=mrate)
            self._aggregateMemberError(m, error, rate)

        # Do a mutation pass?
        if self._shouldMutate(rate):
            self._members.sort(key=lambda m: m.error)
            self._mutate(rate)


    def _aggregateMemberError(self, member, error, rate):
        """Recalculates member.error.
        """
        alpha = 0.2
        if error < member.error:
            alpha = 0.08
        member.setError(alpha * error + (1.0 - alpha) * member.error)


    def _getEliteRate(self, rate):
        """Gets the training rate for an elite node (sorted index < 
        self._eliteCount)."""
        return rate * 0.1


    def _getRouletteMember(self):
        """Gets a random member according to their overall effort.
        """
        maxError = -1e35
        minError = 1e35
        rouletteTotal = 0.0
        for m in self._members:
            maxError = max(m.error, maxError)
            minError = min(m.error, minError)
        for m in self._members:
            rouletteTotal += (maxError - minError) + maxError - m.error
        roulette = random.random() * rouletteTotal
        for m in self._members:
            roulette -= (maxError - minError) + (maxError - m.error)
            if roulette <= 0:
                break
        return m


    def _shouldMutate(self, rate):
        """Returns True if a mutation pass should happen, False otherwise.
        """
        if random.random() >= 1.0 / 7:#rate:
            return False
        return True


    def _mutate(self, rate):
        """Called after our network's members have been re-ordered according
        to each member's fitness.  Performs mutations on poorly performing
        members.
        """
        self._mutateCount += 1

        minError = 1e35
        maxError = -1e35
        for i, m in enumerate(self._members):
            minError = min(minError, m.error)
            maxError = max(maxError, m.error)

        errorTotal = 0.0
        for m in self._members:
            errorTotal += m.error - minError

        for i, m in enumerate(self._members):
            if i < self._eliteCount:
                # elitism
                continue
            if errorTotal <= 1e-35:
                errorTotal = 1.0
            errorFrac = ((m.error - minError) / errorTotal) ** 1.0
            if random.random() < errorFrac * rate:
                # MUTATE AND COPY!!!!
                n = self._getRouletteMember()
                m.mimic(n)
                m.mutate(rate=rate)


class EcoSystemStable(EcoSystem):
    def __init__(self, iters, *args):
        EcoSystem.__init__(self, *args)
        self._trialsPerIter = iters
        self._trialsTillTrain = iters


    def _shouldMutate(self, rate):
        self._trialsTillTrain -= 1
        if self._trialsTillTrain <= 0:
            self._trialsTillTrain = self._trialsPerIter
            return True
        return False


class EcoSystemVar(EcoSystem):
    """Self-adjusting eco system."""
    BACKPROP_RATE = 0
    POPULATION = 1
    ELITE_RATE = 2
    TRIALS_PER = 3
    HEAT_DELTA = 4
    HEAT_MUTATE_MULT = 5
    PERF_ALPHA_WORSE = 6
    PERF_ALPHA_BETTER = 7
    MEMBER_ERROR_AGGR_WORSE = 8
    MEMBER_ERROR_AGGR_BETTER = 9
    LAST_ERROR_WORSE = 10
    LAST_ERROR_BETTER = 11

    # Both of these are inclusive
    MUTATE_TUNE_MIN = BACKPROP_RATE
    MUTATE_TUNE_MAX = TRIALS_PER

    @property
    def _trialsPerMutate(self):
        return self._values[self.TRIALS_PER]

    @property
    def _mutatesPerOptimize(self):
        return 1.0 #self._values[self.TRIALS_PER]

    @property
    def _mutateRate(self):
        """Multiplier for % of error that this member is to receive a 
        mutation.
        """
        return self._values[self.HEAT_MUTATE_MULT] * self._heat

    def __init__(self, logicClass, values = None):
        defaultValues = [ 0.02068, 5.3765, 0.3599, 1.0,
                    1.142, 0.998,
                    0.00934, 0.23958,
                    0.4498, 0.13673,
                    0.8928, 0.02161,
                    ]
        if values is None:
            self._values = defaultValues
        else:
            self._values = values[:] + defaultValues[len(values):]

        EcoSystem.__init__(self, logicClass, int(self._values[self.POPULATION]))

        self._minPopulation = 1
        self._maxPopulation = 1
        self._oldValues = self._values
        self._lastError = None
        self._lastPerf = None
        self._trialsTillMutate = self._trialsPerMutate
        self._mutatesTillOptimize = self._mutatesPerOptimize
        self._heatMin = 0.0001
        self._heatMax = 0.8
        self._heat = self._heatMax # Percentage change in a value; start out
                                   # volatile


    def train(self, inputs, outputs):
        EcoSystem.train(self, inputs, outputs, 
                rate = self._values[self.BACKPROP_RATE])


    def _aggregateMemberError(self, member, error, rate):
        """Aggregate"""
        alpha = self._values[self.MEMBER_ERROR_AGGR_WORSE]
        if error < member.error:
            alpha = self._values[self.MEMBER_ERROR_AGGR_BETTER]
        member.setError(alpha * error + (1 - alpha) * member.error)


    def _getEliteRate(self, rate):
        return rate * self._values[self.ELITE_RATE]


    def _shouldMutate(self, rate):
        self._trialsTillMutate -= 1.0
        if self._trialsTillMutate <= 0:
            return True
        return False


    def _mutate(self, rate):
        # Find out our performance this time (we want to minimize perf)
        newError = self._members[0].error
        if self._lastError is None:
            self._lastError = newError

        # Since we add or subtract in the range of [-50%, 50%], numbers have
        # a natural inclination to go downward.  So population will 
        # automatically optimize downward.
        perf = (newError / self._lastError) * (1.0 + len(self._members) * 0.015)
        # * len(self._members)

        self._mutatesTillOptimize -= 1.0
        if self._mutatesTillOptimize <= 0:
            self._mutatesTillOptimize = self._mutatesPerOptimize

            if self._lastPerf is None:
                self._lastPerf = perf
            delta = perf - self._lastPerf
            h = 1e-35 # ITS ICE COLD IN HERE

            # alpha is the performance aggregation value, used to weight the
            # new performance against the weighted average of past
            # performances
            alpha = 0.0

            if delta > 0.0:
                self._values = self._oldValues

                # We don't want to completely go to the new perf, since
                # after all, it performed worse.
                alpha = self._values[self.PERF_ALPHA_WORSE]
            else:
                # We're keeping the new values
                alpha = self._values[self.PERF_ALPHA_BETTER]

            heatChange = self._values[self.HEAT_DELTA]
            # We expect a small improvement to cool down the system
            if newError < self._lastError:
                self._heat /= heatChange * heatChange
            else:
                self._heat *= heatChange
            #self._heat *= (newError / self._lastError)
            self._heat = max(self._heatMin, min(self._heatMax, self._heat))

            self._lastPerf = (1.0 - alpha) * self._lastPerf + alpha * perf
            ea = self._values[self.LAST_ERROR_WORSE]
            if newError < self._lastError:
                # If this round is better, then converge faster _lastError
                # faster so that we don't cool off too much
                ea = self._values[self.LAST_ERROR_BETTER]
            self._lastError = newError * ea + (1.0 - ea) * self._lastError

            # Generate a next try
            self._oldValues = self._values[:]
            val = random.randint(self.MUTATE_TUNE_MIN, self.MUTATE_TUNE_MAX)
            minAbsChange = 0.0
            if val in [ self.POPULATION, self.TRIALS_PER ]:
                minAbsChange = 1.0
            valDelta = (self._values[val] * self._heat 
                    * (random.random() * 2.0 - 1.0))
            if abs(valDelta) < minAbsChange:
                if valDelta >= 0:
                    valDelta = minAbsChange
                else:
                    valDelta = -minAbsChange
            self._values[val] += valDelta
            
            # Every X generations, have a "dramatic" event
            if random.random() < 0.08:
                a = self._values[self.POPULATION]
                if random.random() > 0.4:
                    # New birth!
                    if a < 10:
                        a += 8
                    else:
                        a *= 2
                else:
                    # Plague
                    if a > 4:
                        a /= 2
                    else:
                        a = min(a, 2)
                self._values[self.POPULATION] = a

            self._values[self.POPULATION] = max(self._minPopulation,
                    self._values[self.POPULATION])
            self._values[self.TRIALS_PER] = max(1.0,
                    self._values[self.TRIALS_PER])

            self._tempCount = getattr(self, '_tempCount', 0) + 1
            if self._tempCount >= 210 and True:
                self._tempCount = 0
                print("VALUES: {0}/{2}/{1}".format(newError, self._values, 
                        self._heat))

        # Adjust population and training according to new params
        self._trialsTillMutate = self._trialsPerMutate

        newPop = min(self._maxPopulation, max(self._minPopulation,
                int(self._values[self.POPULATION])))
        while newPop < len(self._members):
            self.killSomeone()

        # Mutate existing population before inserting new population, otherwise
        # we might get uninitialized neural nets.
        EcoSystem._mutate(self, self._mutateRate)

        while newPop > len(self._members):
            self.addSomeone(rate)


