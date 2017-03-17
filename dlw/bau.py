import numpy as np
from abc import ABCMeta, abstractmethod

class BusinessAsUsual(object):
    """Abstract BAU class for the DLW-model."""
    __metaclass__ = ABCMeta
    def __init__(self, ghg_start, ghg_end, emit_time, emit_level):
        self.ghg_start = ghg_start
        self.ghg_end = ghg_end
        self.emit_time = emit_time
        self.emit_level = emit_level
        self.emission_by_decisions = None
        self.emission_per_period = None
        self.emission_to_ghg = None
        self.emission_to_bau = None
        self.bau_path = None

    @abstractmethod
    def emission_by_time(self):
        pass


class DLWBusinessAsUsual(BusinessAsUsual):
    """Business-as-usual scenario of emissions. Emissions growth is assumed to 
    slow down exogenously - these assumptions represent an attempt to model 
    emissions growth in a business-as-usual scenario that is in the absence of incentives.

    Parameters:
        ghg_start (float): Start value of GHG level.
        ghg_end (float): Assumed GHG level at the end of last period.
        emit_level (list): Emission levels in future times.
        emit_time (list): Times of given emission levels 'emit_level'.
        
    """

    def __init__(self, ghg_start=400.0, ghg_end=1000.0, emit_time=[0, 30, 60], emit_level=[52.0, 70.0, 81.4]):
        super(DLWBusinessAsUsual, self).__init__(ghg_start, ghg_end, emit_time, emit_level)

    def emission_by_time(self, time):
        """
        Return the BAU emissions at any time.
        """
        if time < self.emit_time[1]:
            emissions = self.emit_level[0] + float(time) / (self.emit_time[1] - self.emit_time[0]) \
                        * (self.emit_level[1] - self.emit_level[0])
        elif time < self.emit_time[2]:
            emissions = self.emit_level[1] + float(time - self.emit_time[1]) / (self.emit_time[2] 
                        - self.emit_time[1]) * (self.emit_level[2] - self.emit_level[1])
        else:
            emissions = self.emit_level[2]
        return emissions

    def bau_emissions_setup(self, tree):
        """
        Create default business as usual emissions path.
        The emissions rate in each period are assumed to be the average of the emissions 
        at the beginning and at the end of the period.

        Args:
            tree (obj 'TreeModel'): DLW tree-model.

        """
        num_periods = tree.num_periods
        self.emission_by_decisions = np.zeros(num_periods)
        self.emission_per_period = np.zeros(num_periods)
        #self.emissions_to_ghg = np.zeros(num_periods)
        self.bau_path = np.zeros(num_periods)
        self.bau_path[0] = self.ghg_start
        self.emission_by_decisions[0] = self.emission_by_time(tree.decision_times[0])
        period_len = tree.decision_times[1:] - tree.decision_times[:-1]

        for n in range(1, num_periods):
            self.emission_by_decisions[n] = self.emission_by_time(tree.decision_times[n])
            self.emission_per_period[n] = period_len[n] * (self.emission_by_decisions[n-1:n].mean())

        #the total increase in ghg level of 600 (from 400 to 1000) in the bau path is allocated over time
        self.emission_to_ghg = (self.ghg_end - self.ghg_start) * self.emission_per_period / self.emission_per_period.sum()
        self.emission_to_bau = self.emission_to_ghg[-1] / self.emission_per_period[-1]
        for n in range(1, num_periods):
            self.bau_path[n] = self.bau_path[n-1] + self.emission_per_period[n]*self.emission_to_bau



