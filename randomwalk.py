class RandomWalk:
    def __init__(self):
        # start from the middle node by default,
        # can be set by the algorithm if exploration start is needed.
        self.initial = 3
        self._reward = 1
        self._current = None
        self._done = False
        self.nA = 2
        self.nS = 7

    def seed(self,seed):
        pass

    def reset(self):
        self._current = self.initial
        self._done = False
        return self._current

    def step(self,action):
        if self._current == 0 or self._current == 6:
            self._done = True
            return self._current, 0, self._done, None
        if action == 0:
            self._current -= 1
        else:
            self._current += 1
        if self._current == 0:
            self._done = True
            return self._current, 0, self._done, None
        elif self._current == 6:
            self._done = True
            return self._current, self._reward, self._done, None
        else:
            return self._current, 0, self._done, None
