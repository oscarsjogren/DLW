import numpy as np
from abc import ABCMeta, abstractmethod
from storage_tree import BigStorageTree, SmallStorageTree
import multiprocessing
from tools import _pickle_method, _unpickle_method
import copy_reg
import types

np.seterr(all='ignore')
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Utility(object):
	"""Abstract utility class for the DLW-model."""
	__metaclass__ = ABCMeta

	def __init__(self, tree, damage, cost, period_len):
		self.tree = tree
		self.damage = damage
		self.cost = cost
		self.period_len = period_len
		self.decision_times = tree.decision_times
		self.cons_growth = damage.cons_growth
		self.growth_term = 1.0 + self.cons_growth

	@abstractmethod
	def utility(self):
		pass

	@abstractmethod
	def adjusted_utility(self):
		pass

	@abstractmethod
	def marginal_utility(self):
		pass

	@abstractmethod
	def numerical_gradient(self):
		pass

class EZUtility(Utility):
	"""Calculation of Epstein-Zin utility for the DLW-model.

	Parameters:
		tree (obj 'TreeModel'): Provides the tree structure used.
		damage (obj 'Damage'): Provides methods for calculating damages.
		cost (obj 'Cost'): Provides methods for calculating cost of mitigation.
		period_len (int): The length of the sub-intervals.
		eis (float): Elasticity of intertemporal substitution.
		ra (float): Risk-aversion.
		time_pref (float): Pure rate of time preference.

	"""

	def __init__(self, tree, damage, cost, period_len, eis=0.9, ra=7.0, time_pref=0.005):
		super(EZUtility, self).__init__(tree, damage, cost, period_len)
		self.eis = eis
		self.ra = ra
		self.time_pref = time_pref
		self.r = 1.0 - 1.0/eis
		self.a = 1.0 - ra
		self.b = (1.0-time_pref)**period_len
		self.potential_cons = np.ones(self.decision_times.shape) + self.cons_growth
		self.potential_cons = self.potential_cons ** self.decision_times

	def _end_period_utility(self, m, utility_tree, cons_tree, cost_tree):
		"""Calculate the terminal utility."""
		period_ave_mitigation = self.damage.average_mitigation(m, self.tree.num_periods)
		period_damage = self.damage.damage_function(m, self.tree.num_periods)
		damage_nodes = self.tree.get_nodes_in_period(self.tree.num_periods)
		
		period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]
		period_cost = self.cost.cost(self.tree.num_periods, period_mitigation, period_ave_mitigation)
		continuation = (1.0 / (1.0 - self.b*(self.growth_term**self.r)))**(1.0/self.r)

		cost_tree.set_value(cost_tree.last_period, period_cost)
		cons_tree.set_value(cons_tree.last_period, self.potential_cons[-1] * (1.0 - period_damage))
		utility_tree.set_value(utility_tree.last_period, (1.0 - self.b)**(1.0/self.r) * cons_tree.last * continuation)

	def _end_period_marginal_utility(self, mu_tree_0, mu_tree_1, ce_tree, utility_tree, cons_tree):
		"""Calculate the terminal marginal utility."""
		ce_term = utility_tree.last**self.r - (1.0 - self.b)*cons_tree.last**self.r
		ce_tree.set_value(ce_tree.last_period, ce_term)

		mu_0_last = (1.0 - self.b)*(utility_tree[utility_tree.last_period-self.period_len] / cons_tree.last)**(1.0-self.r)
		mu_tree_0.set_value(mu_tree_0.last_period, mu_0_last)
		mu_0 = self._mu_0(cons_tree[cons_tree.last_period-self.period_len], ce_tree[ce_tree.last_period-self.period_len])
		mu_tree_0.set_value(mu_tree_0.last_period-self.period_len, mu_0)

		next_term = self.b * (1.0 - self.b) / (1.0 - self.b * self.growth_term**self.r)
		mu_1 = utility_tree[utility_tree.last_period-self.period_len]**(1-self.r) * next_term * cons_tree.last**(self.r-1.0)
		mu_tree_1.set_value(mu_tree_1.last_period-self.period_len, mu_1)
		

	def _certain_equivalence(self, period, damage_period, utility_tree):
		"""Caclulate certainty equivalence utility. If we are between decision nodes, i.e. no branching,
		then certainty equivalent utility at time period depends only on the utility next period 
		given information known today. Otherwise the certainty equivalent utility is the ability 
		weighted sum of next period utility over the partition reachable from the state.
		"""
		if utility_tree.is_information_period(period):
			damage_nodes = self.tree.get_nodes_in_period(damage_period+1)
			probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1]+1]
			even_probs = probs[::2]
			odd_probs = probs[1::2]
			even_util = ((utility_tree.get_next_period_array(period)[::2])**self.a) * even_probs
			odd_util = ((utility_tree.get_next_period_array(period)[1::2])**self.a) * odd_probs
			ave_util = (even_util + odd_util) / (even_probs + odd_probs)
			cert_equiv = ave_util**(1.0/self.a)
		else:
			# no branching implies certainty equivalent utility at time period depends only on
			# the utility next period given information known today
			cert_equiv = utility_tree.get_next_period_array(period)
		return cert_equiv

	def _utility_generator(self, m, utility_tree, cons_tree, cost_tree, ce_tree, cons_adj=0.0):
		"""Generator for calculating utility for each utility period besides the terminal utility."""
		periods = utility_tree.periods[::-1]
		for period in periods[1:]:
			damage_period = utility_tree.between_decision_times(period)
			cert_equiv = self._certain_equivalence(period, damage_period, utility_tree)

			if utility_tree.is_decision_period(period+self.period_len):
				damage_nodes = self.tree.get_nodes_in_period(damage_period)	
				period_mitigation =  m[damage_nodes[0]:damage_nodes[1]+1]
				period_ave_mitigation = self.damage.average_mitigation(m, damage_period)
				period_cost = self.cost.cost(damage_period, period_mitigation, period_ave_mitigation)
				period_damage = self.damage.damage_function(m, damage_period)
				cost_tree.set_value(cost_tree.index_below(period+self.period_len), period_cost)

			
			# should this be (1.0 - period_damage - period_cost)?
			period_consumption = self.potential_cons[damage_period] * (1.0 - period_damage) * (1.0 - period_cost)
			if period == 0:
				period_consumption += cons_adj
		
			if not utility_tree.is_decision_period(period):
				next_consumption = cons_tree.get_next_period_array(period)
				segment = period - utility_tree.decision_times[damage_period]
				interval = segment + utility_tree.subinterval_len

				if utility_tree.is_decision_period(period+self.period_len):
					next_cost = cost_tree[period+self.period_len]
					if period < utility_tree.decision_times[-2]:
						next_consumption *= (1.0 - np.repeat(period_cost,2)) /(1.0 - next_cost)
					# don't we want to do this also for the 'straight' periods?
					#else:
					#	next_consumption *= (1.0 - period_cost) / (1.0 - next_cost)
					#	
						
				if period < utility_tree.decision_times[-2]:
					period_consumption = ((next_consumption/np.repeat(period_consumption,2))**(segment/float(interval))) * np.repeat(period_consumption,2)
				else:
					period_consumption = ((next_consumption/period_consumption)**(segment/float(interval)))* period_consumption
				
			ce_term = self.b * cert_equiv**self.r
			ce_tree.set_value(period, ce_term)
			cons_tree.set_value(period, period_consumption)
			u = ((1.0-self.b)*period_consumption**self.r + ce_term)**(1.0/self.r)
			u[np.where(np.isnan(u))] = 0.0 	# get nan-values when negative 
			yield u, period

	def utility(self, m, return_trees=False):
		"""Calculating utility for the specific mitigation decisions 'm'.

		Args:
			m (ndarray): Array of mitigations.
			return_trees (bool): True if method should return trees calculculated in producing the utility.

		Returns:
			tuple of ndarrays if return_trees else float of period 0's utility.

		"""
		# can we make this smarter and not create these every time we call the utility?
		utility_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cons_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		ce_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cost_tree = SmallStorageTree(decision_times=self.decision_times)

		self._end_period_utility(m, utility_tree, cons_tree, cost_tree)
		it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree)
		for u, period in it:
			utility_tree.set_value(period, u)
			
		if return_trees:
			return utility_tree, cons_tree, cost_tree, ce_tree
		return utility_tree[0]


	def adjusted_utility(self, m, period_cons_eps=None, node_cons_eps=None, final_cons_eps=0.0,
						 first_period_consadj=0.0, return_trees=False):
		"""Calculating adjusted utility for sensitivity analysis. Used e.g. to find zero-coupon bond price.

		Args:
			m (ndarray): 1D-array of mitigations.
			period_cons_eps (optionla, ndarray): Array of increases in consumption per period. #77
			node_cons_eps (optional, ndarray): Array of increases in consumption per node. #(big_storage)
			final_cons_eps (optional, float): Number to increase the final utility.
			first_period_consadj (optional, float): Adjustment of consumption at time 0.
			return_trees (bool): True if method should return trees calculculated in producing the utility.

		Returns:
			tuple of ndarrays if return_trees else float of period 0's utility.

		"""
		# can we make this smarter and not create these every time we call the utility?
		utility_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cons_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		ce_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		cost_tree = SmallStorageTree(decision_times=self.decision_times)
		
		self._end_period_utility(m, utility_tree, cons_tree, cost_tree)
		periods = utility_tree.periods[::-1]

		if period_cons_eps is None:
			period_cons_eps = np.zeros(len(periods))
		if node_cons_eps is None:
			node_cons_eps = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)

		it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree, first_period_consadj)
		i = len(utility_tree)-2
		for u, period in it:
			if period == periods[1]:
				mu_0 = (1.0 - self.b)*(u / cons_tree[period])**(1.0-self.r)
				next_term = self.b * (1.0 - self.b) / (1.0 - self.b * self.growth_term**self.r)
				mu_1 = (u**(1.0-self.r)) * next_term * (cons_tree.last**(self.r-1.0))
				u += (final_cons_eps + period_cons_eps[-1]) * mu_1
				u +=  (period_cons_eps[i] + node_cons_eps.tree[period])*mu_0
				utility_tree.set_value(period, u)
			else:
				mu_0, m_1, m_2 = self._period_marginal_utility(mu_0, mu_1, m, period, utility_tree, cons_tree, ce_tree)
				u +=  (period_cons_eps[i] + node_cons_eps.tree[period])*mu_0
				utility_tree.set_value(period, u)
			i -= 1

		if return_trees:
			return utility_tree, cons_tree, cost_tree, ce_tree
		return utility_tree.tree[0]

	def _mu_0(self, cons, ce_term):
		"""Marginal utility with respect to consumption function."""
		t1 = (1.0 - self.b)*cons**(self.r-1.0)
		t2 = (ce_term - (self.b-1.0)*cons**self.r)**((1.0/self.r)-1.0)
		return t1 * t2

	def _mu_1(self, cons, prob, cons_1, cons_2, ce_1, ce_2, do_print=False):
		""" marginal utility with respect to consumption next period."""
		t1 = (1.0-self.b) * self.b * prob * cons_1**(self.r-1.0)
		t2 = (ce_1 - (self.b-1.0) * cons_1**self.r )**((self.a/self.r)-1)
		t3 = (prob * (ce_1 - (self.b*(cons_1**self.r)) + cons_1**self.r)**(self.a/self.r) \
			 + (1.0-prob) * (ce_2 - (self.b-1.0) * cons_2**self.r)**(self.a/self.r))**((self.r/self.a)-1.0)
		t4 = prob * (ce_1-self.b * (cons_1**self.r) + cons_1**self.r)**(self.a/self.r) \
			 + (1.0-prob) * (ce_2 - self.b * (cons_2**self.r) + cons_2**self.r)**(self.a/self.r)
		t5 = (self.b * t4**(self.r/self.a) - (self.b-1.0) * cons**self.r )**((1.0/self.r)-1.0)

		return t1 * t2 * t3 * t5 

	def _mu_2(self, cons, prev_cons, ce_term):
		"""Marginal utility with respect to last period consumption."""
		t1 = (1.0-self.b) * self.b * prev_cons**(self.r-1.0)
		t2 = ((1.0 - self.b) * cons**self.r - (self.b - 1.0) * self.b \
		     * prev_cons**self.r + self.b * ce_term)**((1.0/self.r)-1.0)
		return t1 * t2

	def _period_marginal_utility(self, prev_mu_0, prev_mu_1, m, period, utility_tree, cons_tree, ce_tree):
		"""Marginal utility for each node in a period."""
		damage_period = utility_tree.between_decision_times(period)
		mu_0 = self._mu_0(cons_tree[period], ce_tree[period])
		
		prev_ce = ce_tree.get_next_period_array(period)
		prev_cons = cons_tree.get_next_period_array(period)
		if utility_tree.is_information_period(period):
			probs = self.tree.get_probs_in_period(damage_period+1)
			up_prob = np.array([probs[i]/(probs[i]+probs[i+1]) for i in range(0, len(probs), 2)])
			down_prob = 1.0 - up_prob

			up_cons = prev_cons[::2]
			down_cons = prev_cons[1::2]
			up_ce = prev_ce[::2]
			down_ce = prev_ce[1::2]
			
			mu_1 = self._mu_1(cons_tree[period], up_prob, up_cons, down_cons, up_ce, down_ce)
			mu_2 = self._mu_1(cons_tree[period], down_prob, down_cons, up_cons, down_ce, up_ce)
			# not optimal
			return mu_0, mu_1, mu_2 
		else:
			mu_1 = self._mu_2(cons_tree[period], prev_cons, prev_ce)
			# not optimal
			return mu_0, mu_1, None 

	def marginal_utility(self, m, utility_tree, cons_tree, cost_tree, ce_tree):
		"""Calculating marginal utility for sensitivity analysis, e.g. in the SSC decomposition.

		Args:
			m (ndarray): 1D-array of mitigations.
			period_cons_eps (optionla, ndarray): Array of increases in consumption per period. #77
			node_cons_eps (optional, ndarray): Array of increases in consumption per node. #(big_storage)
			final_cons_eps (optional, float): Number to increase the final utility.
			first_period_consadj (optional, float): Adjustment of consumption at time 0.
			return_trees (bool): True if method should return trees calculculated in producing the utility.

		Returns:
			tuple of ndarrays if return_trees else float of period 0's utility.

		"""
		#could add ce_tree to parameter list.
		mu_tree_0 = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		mu_tree_1 = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		mu_tree_2 = SmallStorageTree(decision_times=self.decision_times)
		#ce_tree = BigStorageTree(subinterval_len=self.period_len, decision_times=self.decision_times)
		self._end_period_marginal_utility(mu_tree_0, mu_tree_1, ce_tree, utility_tree, cons_tree)
		periods = utility_tree.periods[::-1]

		for period in periods[2:]:
			mu_0, mu_1, mu_2 = self._period_marginal_utility(mu_tree_0.get_next_period_array(period),
				mu_tree_1.get_next_period_array(period), m, period, utility_tree, cons_tree, ce_tree)
			mu_tree_0.set_value(period, mu_0)
			mu_tree_1.set_value(period, mu_1)
			if mu_2 is not None:
				mu_tree_2.set_value(period, mu_2)

		return mu_tree_0, mu_tree_1, mu_tree_2



	def _grad_helper(self, i):
		m_copy = self.m.copy()
		m_copy[i] -= self.delta
		minus_utility = self.utility(m_copy)
		m_copy[i] += 2*self.delta
		plus_utility = self.utility(m_copy)
		grad = (plus_utility-minus_utility) / (2*self.delta)
		return grad, i

	def numerical_gradient(self, m, delta=1e-06):
		self.delta = delta
		self.m = m 
		grad = np.zeros(len(m))
		if not isinstance(m, np.ndarray):
			self.m = np.array(m)

		pool = multiprocessing.Pool()
		res = pool.map(self._grad_helper, range(len(m)))
		for g, i in res:
			grad[i] = g
		pool.close()
		pool.join()
		del self.m
		del self.delta
		return grad


	
