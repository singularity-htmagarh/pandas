import numpy as np


class EWMean:

    def __init__(self, com, adjust, ignore_na):
        alpha = 1. / (1. + com)
        self.old_wt_factor = 1. - alpha
        self.new_wt = 1. if adjust else alpha
        self.old_wt = 1.
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.weighted_avg = None

    def step(self, observation, delta=1):
        if self.weighted_avg is not None:
            if not observation.isnull().all() or not self.ignore_na:
                self.old_wt *= self.old_wt_factor ** delta
                if not observation.isnull().all():
                    self.weighted_avg = ((self.old_wt * self.weighted_avg) + (self.new_wt * observation)) / (self.old_wt + self.new_wt)
                if self.adjust:
                    self.old_wt += self.new_wt
                else:
                    self.old_wt = 1.
        elif not observation.isnull().all():
            self.weighted_avg = observation

    def finalize(self):
        return self.weighted_avg

    def reset(self):
        self.old_wt = 1
        self.weighted_avg = np.nan
