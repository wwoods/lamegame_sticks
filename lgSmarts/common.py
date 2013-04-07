
class NamedLinearArray(object):
    """Maps from a dict to a linear, predictably indexed array.
    """

    def __init__(self):
        """Returns the addresses"""
        self._addresses = {}
        self._names = []


    def __len__(self):
        return len(self._names)


    def convertDict(self, d):
        """Returns an array representing the values of d (values not supplied
        are filled in with None).  For keys in d that do not already have 
        addresses, one is created.
        """
        result = [ None ] * len(self._names)
        for k, v in d.items():
            idx = self._addresses.get(k)
            if idx is None:
                self._addresses[k] = idx = len(self._names)
                self._names.append(k)
                result.append(v)
            else:
                result[idx] = v
        return result


    def convertArray(self, v):
        """Returns a dict representing the values of v (values that are None
        in v will not have output keys).
        """
        result = {}
        for i, d in enumerate(v):
            if d is None:
                continue
            k = self._names[i]
            result[k] = d
        return result


