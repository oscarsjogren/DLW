import numpy as np
from abc import ABCMeta, abstractmethod

class Cost(object):
	"""Abstract Cost class for the DLW-model."""
	__metaclass__ = ABCMeta

	@abstractmethod
	def cost(self):
		pass

	@abstractmethod
	def price(self):
		pass


class DLWCost(Cost):
	"""Class to evaluate the cost curve for the DLW-model.
	Parameters:
		tree (obj: 'TreeModel'): Provides the tree structure used.
		emit_at_0 (float): Initial GHG emission level.
		g (float): Intital scale of the cost function.
		a (float): Curvature of the cost function.
		join_price (float): Price at which the cost curve is extended.
		max_price (float): Price at which carbon dioxide can be removed from 
			atmosphere in unlimited scale.
		tech_const (float): Determines the degree of exogenous technological improvement 
			over time. A number of 1.0 implies 1 percent per yer lower cost.
		tech_scale (float): Determines the sensitivity of technological change 
			to previous mitigation. 
		cons_at_0 (float): Intital consumption. Default $30460bn based on US 2010 values.
	"""

	def __init__(self, tree, emit_at_0, g, a, join_price, max_price,
				tech_const, tech_scale, cons_at_0):
		self.tree = tree
		self.g = g
		self.a = a
		self.join_price = join_price
		self.max_price = max_price
		self.tech_const = tech_const
		self.tech_scale = tech_scale
		self.cbs_level = (join_price / (g * a))**(1.0 / (a - 1.0))
		self.cbs_deriv = self.cbs_level / (join_price * (a - 1.0))
		self.cbs_b = self.cbs_deriv * (max_price - join_price) / self.cbs_level
		self.cbs_k = self.cbs_level * (max_price - join_price)**self.cbs_b
		self.cons_per_ton = cons_at_0 / emit_at_0
		self.cost_gradient = np.zeros((tree.num_decision_nodes, tree.num_decision_nodes))

	def cost_by_state(self, node, mitigation, ave_mitigation):
		"""Calculates the mitigation cost by state.
		Args:
			node (int): Node in tree for which mitigation cost is calculated.
			mitigation (float): Current mitigation value
			ave_mitigation (float): Average mitigation per year up to this point.
		Returns:
			float: Cost by state (cbs)
		"""

		period = self.tree.get_period(node)
		years = self.tree.decision_times[period]

		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100))**years
		if mitigation < self.cbs_level:
			cbs = self.g * (mitigation**self.a) * tech_term / self.cons_per_ton
		else:
			base_cbs = self.g * self.cbs_level**self.a
			extension = ((mitigation-self.cbs_level) * self.max_price 
						 - self.cbs_b*mitigation * (self.cbs_k/mitigation)**(1.0/self.cbs_b) / (self.cbs_b-1.0)
						 + self.cbs_b*self.cbs_level * (self.cbs_k/self.cbs_leve)**(1.0/self.cbs_b)/(self.cbs_b-1.0))
			cbs = (base_cbs + extension) * tech_term / self.cons_per_ton
		return cbs

	def cost(self, period, mitigation, ave_mitigation):
		"""Calculates the mitigation cost by state.
		Args:
			period (int): Period in tree for which mitigation cost is calculated.
			mitigation (ndarray): Current mitigation values for period
			ave_mitigation (ndarray): Average mitigation per year up to this point for all 
				nodes in the period.
		Returns:
			ndarray: Cost by state (cbs)
		"""
		years = self.tree.decision_times[period]
		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100.0))**years
		cbs = self.g * (mitigation**self.a) 
		bool_arr = (mitigation < self.cbs_level).astype(int)
		if np.all(bool_arr):
			return cbs * tech_term / self.cons_per_ton

		base_cbs = self.g * self.cbs_level**self.a
		bool_arr2 = (mitigation > self.cbs_level).astype(int)
		extension = ((mitigation-self.cbs_level) * self.max_price 
					- self.cbs_b*mitigation * (self.cbs_k/mitigation)**(1.0/self.cbs_b) / (self.cbs_b-1.0)
					+ self.cbs_b*self.cbs_level * (self.cbs_k/self.cbs_level)**(1.0/self.cbs_b) / (self.cbs_b-1.0))
		
		c = (cbs * bool_arr + (base_cbs + extension)*bool_arr2) * tech_term / self.cons_per_ton
		c = np.nan_to_num(c) # we might have nan values that should be set to zero
		return c

	def price(self, years, mitigation, ave_mitigation):
		"""Inverse of the cost function. Gives emissions price for any given 
		degree of mitigation, average_mitigation, and horizon.
		Args:
			years (int): Years of technological change so far.
			mitigation (float): Current mitigation value.
			ave_mitigation (float): Average mitigation per year up to this point.
		Returns:
			float: The price.
		"""
		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100))**years
		if mitigation < self.cbs_level:
			return self.g * self.a * (mitigation**(self.a-1.0)) * tech_term
		else:
			return (self.max_price - (self.cbs_k/mitigation)**(1.0/self.cbs_b)) * tech_term

	def price_by_state_period(self, period, mitigation, ave_mitigation):
		"""Inverse of the cost function. Gives emissions price for any given 
		degree of mitigation, average_mitigation, and horizon.
		Args:
			period (int): The period for which the price is to be calculated.
			mitigation (ndarray): Current mitigation value.
			ave_mitigation (ndarray): Average mitigation per year up to this point.
		Returns:
			ndarray: The array of prices.
		"""
		years = self.tree.decision_times[period]
		tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation) / 100))**years
		bool_arr = (mitigation < self.cbs_level).astype(int)
		base_price = self.g * self.a * (mitigation**(self.a-1.0))
		if np.all(bool_arr):
			return base_price * tech_term

		bool_arr2 = (mitigation > self.cbs_level).astype(int)
		add_price = (self.max_price - (self.cbs_k/mitigation)**(1.0/self.cbs_b))
		return (base_price*bool_arr + add_price*bool_arr2) * tech_term