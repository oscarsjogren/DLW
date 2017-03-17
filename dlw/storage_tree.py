import numpy as np
from abc import ABCMeta, abstractmethod

class BaseStorageTree(object):
	"""Abstract storage class for the DLW-model."""
	__metaclass__ = ABCMeta

	def __init__(self, decision_times):
		self.decision_times = np.array(decision_times)
		self.information_times = self.decision_times[:-2]
		self.periods = None
		self.tree = None

	def __len__(self):
		return len(self.tree)

	def __getitem__(self, key):
		if isinstance(key, int) or isinstance(key, float):
			return self.tree.__getitem__(key).copy()
		else:
			raise TypeError('Index must be int, not {}'.format(type(key).__name__))

	def _init_tree(self):
		self.tree = dict.fromkeys(self.periods)
		i = 0
		for key in self.periods:
			self.tree[key] = np.zeros(2**i)
			if key in self.information_times:
				i += 1
	@property
	def last(self):
		return self.tree[self.decision_times[-1]]

	@property
	def last_period(self):
		return self.decision_times[-1]

	@property
	def nodes(self):
		n = 0
		for array in self.tree.values():
			n += len(array)
		return n

	@abstractmethod
	def get_next_period_array(self, period):
		pass

	def write_csv(self, file_name):
		"""Save the entire tree as csv-file."""
		from tools import write_csv_dict
		write_csv_dict(self.tree, file_name)

	def write_decision_times_csv(self, file_name):
		"""Save the data in decision periods as csv-file."""
		from tools import write_csv_2D
		write_csv_dict(self.tree, file_name, self.decision_times[:-1])

	def set_value(self, period, values):
		"""If period is in periods, set the value of element to values (ndarray)."""
		if period not in self.periods:
			print ValueError("Not a valid period")
		if self.tree[period].shape != values.shape:
			raise ValueError("shapes {} and {} not aligned".format(self.tree[period].shape, values.shape))
		self.tree[period] = values

	def is_decision_period(self, time_period):
		"""Checks if time_period is a decision time for mitigation, where
		time_period is the number of years since start.

		Args:
			time_period (int): Time since the start year of the model.

		Returns:
			bool: True if time_period also is a decision time, else False.
		"""
		return time_period in self.decision_times

	def is_real_decision_period(self, time_period):
		"""Checks if time_period is a decision time besides the last period, where
		time_period is the number of years since start.

		Args:
			time_period (int): Time since the start year of the model.

		Returns:
			bool: True if time_period also is a decision time, else False.
		"""
		return time_period in self.decision_times[:-1]

	def is_information_period(self, time_period):
		"""Checks if time_period is a information time for fragility, where
		time_period is the number of years since start.

		Args:
			time_period (int): Time since the start year of the model.

		Returns:
			bool: True if time_period also is a information time, else False.

		"""
		return time_period in self.information_times

class SmallStorageTree(BaseStorageTree):

	def __init__(self, decision_times):
		super(SmallStorageTree, self).__init__(decision_times)
		self.periods = self.decision_times
		self._init_tree()

	def get_next_period_array(self, period):
		"""Returns the array of the next decision period."""
		if self.is_real_decision_period(period):
			index = self.decision_times[np.where(self.decision_times==period)[0]+1][0]
			return self.tree[index].copy()
		raise IndexError("Given period is not in decision times")

	def index_below(self, period):
		"""Returns the key of the previous decision period."""
		period = self.decision_times[np.where(self.decision_times==period)[0]-1]
		return period[0]

class BigStorageTree(BaseStorageTree):
	"""Storage tree class for the DLW-model.

	Attributes:
		subinterval_len (int): Lenght of subintervals between nodes.
		decision_times (ndarray): List of times where decisions are taken.
		information_times (ndarray): List of times where agent receives new information.
		tree: (ndarray): 3D-array of stored values

	Args:
		subinterval_len (int): Lenght of subintervals between nodes where utility is calculated.
		decision_times (ndarray): List of times where decisions of mitigation is taken.

	"""

	def __init__(self, subinterval_len, decision_times):
		super(BigStorageTree, self).__init__(decision_times)
		self.subinterval_len = subinterval_len
		self.periods = np.arange(0, self.decision_times[-1]+self.subinterval_len,
							 self.subinterval_len)
		self._init_tree()

	def first_period_intervals(self):
		"""Returns the number of subintervals in the first period."""
		return int((self.decision_times[1] - self.decision_times[0]) / self.subinterval_len)

	def get_next_period_array(self, period):
		"""Returns the array of the next period."""
		return self.tree[period+self.subinterval_len].copy()

	def between_decision_times(self, period):
		"""Check which decision time the period is between.

		Examples:
			>>> bst = BigStorageTree(5, [0, 15, 45, 85, 185, 285, 385])
			>>> bst.between_decision_times(5)
			0
			>>> bst.between_decision_times(15)
			1

		"""
		if period == 0:
			return 0
		for i in range(len(self.information_times)):
			if self.decision_times[i] <= period and period < self.decision_times[i+1]:
				return i
		return i+1

	def decision_interval(self, period):
		"""Check which interval the period is between.

		Examples:
			>>> bst = BigStorageTree(5, [0, 15, 45, 85, 185, 285, 385])
			>>> bst.decision_interval(5)
			1
			>>> bst.between_decision_times(15)
			1
			>>> bst.between_decision_times(20)
			2

		"""
		if period == 0:
			return 0
		for i in range(1, len(self.decision_times)):
			if self.decision_times[i-1] < period and period <= self.decision_times[i]:
				return i
		return i

	def get_real_information_array(self): # not used?
		arrays = []
		for period in self.decision_times[:-1]:
			arrays.extend(self.tree[period])
		return np.array(arrays)
