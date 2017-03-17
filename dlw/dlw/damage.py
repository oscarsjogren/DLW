import numpy as np
from abc import ABCMeta, abstractmethod
from damage_simulation import DamageSimulation
from forcing import Forcing

class Damage(object):
	"""Abstract damage class for the DLW-model.

	Parameters:
		tree (obj: 'TreeModel'): Provides the tree structure used.
		bau (obj: 'BusinessAsUsual'): Business-as-usual scenario of emissions.

	"""
	__metaclass__ = ABCMeta
	def __init__(self, tree, bau):
		self.tree = tree
		self.bau = bau

	@abstractmethod
	def average_mitigation(self):
		"""The average_mitigation function should return a 1D array of the
		average mitigation for every node in the period.
		"""
		pass

	@abstractmethod
	def damage_function(self):
		"""The damage_function should return a 1D array of the damages for
		every node in the period.
		"""
		pass

class DLWDamage(Damage):
	"""Damage class for the DLW-model. Provides the damages from emissions and mitigation outcomes.

	Parameters:
		tree (obj: 'TreeModel'): Provides the tree structure used.
		bau (obj: 'BusinessAsUsual'): Business-as-usual scenario of emissions.
		cons_growth (float): Constant consumption growth rate.
		ghg_levels (ndarray or list): End GHG levels for each end scenario.

	"""

	def __init__(self, tree, bau, cons_growth, ghg_levels):
		super(DLWDamage, self).__init__(tree, bau)
		self.ghg_levels = ghg_levels
		if isinstance(self.ghg_levels, list):
			self.ghg_levels = np.array(self.ghg_levels)
		self.cons_growth = cons_growth
		self.dnum = len(ghg_levels)
		self.cum_forcings = None
		self.d = None
		self.forcing = None
		self.damage_coefs = None

	def _damage_interpolation(self):
		"""Create the interpolation coeffiecients used to calculate damages.
		"""
		if self.d is None:
			try:
				print("Importing stored damage simulation")
				self.import_damages()
			except IOError as e:
				import sys
				print("Could not import simulated damages:\n\t{}".format(e))
				sys.exit(0)
		
		self.damage_coefs = np.zeros((self.tree.num_final_states, self.tree.num_periods, self.dnum-1, self.dnum))
		amat = np.ones((self.tree.num_periods, self.dnum, self.dnum))
		bmat = np.ones((self.tree.num_periods, self.dnum))

		self.damage_coefs[:, :, -1,  -1] = self.d[-1, :, :]
											#(last, all, all)
		self.damage_coefs[:, :, -1,  -2] = (self.d[-2, :, :] - self.d[-1, :, :]) / self.emit_pct[-2]
		amat[:, 0, 0] = 2.0 * self.emit_pct[-2]
		amat[:, 1:, 0] = self.emit_pct[:-1]**2
		amat[:, 1:, 1] = self.emit_pct[:-1]
		amat[:, 0, -1] = 0.0

		for state in range(0, self.tree.num_final_states):
			bmat[:, 0] = self.damage_coefs[state, :, -1,  -2] * self.emit_pct[-2]
			bmat[:, 1:] = self.d[:-1, state, :].T
			self.damage_coefs[state, :, 0] = np.linalg.solve(amat, bmat)

	def _forcing_based_mitigation(self, forcing, period): 
		"""Calculation of mitigation based on forcing up to period.

		Args:
			forcing (float): Cumulative forcing up to node.
			period (int): Period of node.

		Returns:
			float: Mitigation.
		"""
		p = period
		if period < self.tree.num_periods-1:
			p -= 1
		if forcing > self.cum_forcings[p][1]:
			weight_on_sim2 = (self.cum_forcings[p][2] - forcing) / (self.cum_forcings[p][2] - self.cum_forcings[p][1])
			weight_on_sim3 = 0
		elif forcing > self.cum_forcings[p][0] :
			weight_on_sim2 = (forcing - self.cum_forcings[p][0]) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
			weight_on_sim3 = (self.cum_forcings[p][1] - forcing) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
		else:
			weight_on_sim2 = 0
			weight_on_sim3 = 1.0 + (self.cum_forcings[p][0] - forcing) / self.cum_forcings[p][0]
		
		return weight_on_sim2 * self.emit_pct[1] + weight_on_sim3*self.emit_pct[0]

	def import_damages(self, loc="data/simulated_damages.csv"):
		with open(loc, 'r') as f:
			d = np.loadtxt(f, delimiter=";", comments="#")
		n = self.tree.num_final_states	
		self.d = np.array([d[n*i:n*(i+1)] for i in range(0, self.dnum)])

	def damage_simulation(self, draws, peak_temp, disaster_tail, tip_on, temp_map, temp_dist_params,
				          maxh, cons_growth, save_simulation=True):					  	  
		"""Initializion of simulation of damages. Either import stored simulation 
		of damages or simulate new values.

		Args:
			import_damages (bool): If program should import already stored values.
				Default is True.
			**kwargs: Arguments to initialize DamageSimulation class, in the
				case of import_damages = False. See DamageSimulation class for 
				more info.

		"""
		ds = DLWDamageSimulation(tree=self.tree, ghg_levels=self.ghg_levels, peak_temp=peak_temp,
					disaster_tail=disaster_tail, tip_on=tip_on, temp_map=temp_map, 
					temp_dist_params=temp_dist_params, maxh=maxh, cons_growth=cons_growth)
		self.d = ds.simulate(draws)

	def forcing_init(self, sink_start, forcing_start, ghg_start, partition_interval, forcing_p1, 
					 forcing_p2, forcing_p3, absorbtion_p1, absorbtion_p2, lsc_p1, lsc_p2): 
		"""Initialize Forcing object and cum_forcings used in calculating 
		the mitigation up to a node. 

		Args:
			**kwargs: Arguments to initialize Forcing object, see
				Forcing class for more info.

		"""
		bau_emission = self.bau.ghg_end - self.bau.ghg_start
		self.emit_pct = 1.0 - (self.ghg_levels-self.bau.ghg_start) / bau_emission
		self.cum_forcings = np.zeros((self.tree.num_periods, self.dnum))
		self.forcing = Forcing(self.tree, self.bau, sink_start, forcing_start, ghg_start, partition_interval,
							forcing_p1, forcing_p2, forcing_p3, absorbtion_p1, absorbtion_p2, lsc_p1, lsc_p2)

		mitigation = np.ones((self.dnum, self.tree.num_decision_nodes)) * self.emit_pct[:, np.newaxis]
		path_ghg_levels = np.zeros((self.dnum, self.tree.num_periods+1))
		path_ghg_levels[0,:] = self.bau.ghg_start

		for i in range(0, self.dnum):
			for n in range(1, self.tree.num_periods+1):
				node = self.tree.get_node(n, 0)
				self.cum_forcings[n-1, i] = self.forcing.forcing_at_node(mitigation[i], node, i)

	def _average_mitigation_node(self, m, node, period):
		"""Calculate the average mitigation until node.

		Args:
			m (ndarray): Array of mitigation.
			node (int): The node for which average mitigation is to be calculated for.

		Returns:
			float: Average mitigation.

		"""
		if period == 0:
			return 0
		period = self.tree.get_period(node)
		state = self.tree.get_state(node, period)
		path = self.tree.get_path(node, period)
		new_m = np.zeros(len(path)-1)
		for i in range(len(new_m)):
			new_m[i] = m[path[i]]
		period_len = self.tree.decision_times[1:period+1] - self.tree.decision_times[:period]
		bau_emissions = self.bau.emission_by_decisions[:period]
		total_emission = np.dot(bau_emissions, period_len)
		ave_mitigation = np.dot(new_m, bau_emissions*period_len)
		return ave_mitigation / total_emission

	def average_mitigation(self, m, period):
		nodes = self.tree.get_num_nodes_period(period)
		ave_mitigation = np.zeros(nodes)
		for i in range(nodes):
			node = self.tree.get_node(period, i)
			ave_mitigation[i] = self._average_mitigation_node(m, node, period)
		return ave_mitigation

	def _damage_function_node(self, m, node):
		"""Calculate the damage at any given node, based on mitigation actions.

		Args:
			m (ndarray): Array of mitigation.
			node (int): The node for which damage is to be calculated for.

		Returns:
			float: damage at node.

		"""
		if self.damage_coefs is None:
			self._damage_interpolation()
		if node == 0:
			return 0.0

		period = self.tree.get_period(node)
		if period >= self.tree.num_periods:
			period = self.tree.num_periods-1

		if period == self.tree.num_periods-2:
			worst_end_state = best_end_state = self.tree.get_state(node, period)
		else:
			worst_end_state, best_end_state = self.tree.reachable_end_states(node, period=period)

		forcing = self.forcing.forcing_at_node(m, node)
		ave_mitigation = self._forcing_based_mitigation(forcing, period)
		probs = self.tree.final_states_prob[worst_end_state:best_end_state+1]

		if ave_mitigation < self.emit_pct[1]:
			damage = (probs *(self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 1] * ave_mitigation \
					 + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 2])).sum()
		
		elif ave_mitigation < self.emit_pct[0]: #do dot product instead?
			damage = (probs * (self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 0] * ave_mitigation**2 \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 1] * ave_mitigation \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 2])).sum()
		
		else: 
			damage = 0.0
			i = 0
			for state in range(worst_end_state, best_end_state+1): 
				if self.d[0, state, period-1] > 10e-4:
					deriv = 2.0 * self.damage_coefs[state, period-1, 0, 0]*self.emit_pct[0] \
							+ self.damage_coefs[state, period-1, 0, 1]
					decay_scale = deriv / (self.d[0, state, period-1]*np.log(0.5))
					dist = ave_mitigation - self.emit_pct[0] + np.log(self.d[0, state, period-1]) \
						   / (np.log(0.5) * decay_scale)
					damage += probs[i] * 0.5**(decay_scale*dist) * np.exp(-np.square(ave_mitigation-self.emit_pct[0])/60.0)
					i += 1

		return damage / probs.sum()

	def damage_function(self, m, period):
		"""Calculate the damage for every node in a period, based on mitigation actions.

		Args:
			m (ndarray): Array of mitigation.
			period (int): The period for which damage is to be calculated.

		Returns:
			ndarray: Array of damages.
		"""
		nodes = self.tree.get_num_nodes_period(period)
		damages = np.zeros(nodes)
		for i in range(nodes):
			node = self.tree.get_node(period, i)
			damages[i] = self._damage_function_node(m, node)
		return damages


if __name__ == "__main__":
	from tree import TreeModel
	from bau import DLWBusinessAsUsual
	from cost import DLWCost
	from damage import DLWDamage
	from utility import EZUtility
	from optimization import GenericAlgorithm, GradientDescent as gd
	from output_functions import ssc_decomposition

	t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0)
	bau_default_model = DLWBusinessAsUsual()
	bau_default_model.bau_emissions_setup(t)
	c = DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=2000.0, max_price=2500.0,
				tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0)
	df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000])
	#df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
	#		temp_map=1, temp_dist_params=None, maxh=100.0, cons_growth=0.015)
	df.import_damages()
	df.forcing_init(sink_start=35.596, forcing_start=4.926, ghg_start=400, partition_interval=5,
		forcing_p1=0.13173, forcing_p2=0.607773, forcing_p3=315.3785, absorbtion_p1=0.94835,
		absorbtion_p2=0.741547, lsc_p1=285.6268, lsc_p2=0.88414)

	m = np.array([0.65008568,0.85875049,0.54217134,1.07955134,0.9877534, 0.77247265, 0.56376919, 
		1.24655422,1.19889565,1.19689472,1.11533683,1.02905754, 0.98775609,0.9220353, 0.40883236, 
		1.24345642,1.2089326, 1.19166262, 1.20695374,1.25336552,1.15316835,1.26275239,1.27616561, 
		1.58004691, 1.54678685,1.54882351,1.54244846,1.61874177,1.55798868,0.97827295,0.92593044, 
		1.12546888,1.37701839,1.28417959,1.03635404,1.26199039,1.36531198,1.08989185,1.11143913, 
		1.28253073,1.1936995, 1.49007705,1.08933526,1.52637337,1.3024672, 1.30407295,1.15306861, 
		1.2353126,1.31761603,1.23053655,1.30587102,1.47995449,1.49003184,1.35051339,1.39986976, 
		1.31363221,1.5914582, 1.62406314,1.48378497,1.66121659,1.49494204,1.44710524,1.20213858])

	u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0)
	utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)

	#ssc_decomposition(m, u, utility_t, cons_t, cost_t, 0.01)
	
	#print "Starting Generic Algorithm \n"
	#ga = GenericAlgorithm(500, 63, 100, 0.80, 0.50, u)
	#m = ga.run(0)

	#print "Moving over to Gradient Descent \n"
	#fixed_values = np.zeros(len(m))
	#fixed_values[0] = m[0]
	#m_hist = gd.run(u, m=m, fixed_values=fixed_values, alpha=0.1, num_iter=10)
	

