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

	def _recombine_nodes(self):
		nperiods = self.tree.num_periods
		sum_class = np.zeros(nperiods, dtype=int)
		new_state = np.zeros([nperiods, self.tree.num_final_states], dtype=int)
		temp_prob = self.tree.final_states_prob.copy()

		for old_state in range(self.tree.num_final_states):
			temp = old_state
			n = nperiods-2
			d_class = 0
			while n >= 0:
				if temp >= 2**n:
					temp -= 2**n
					d_class += 1
				n -= 1
			sum_class[d_class] += 1
			new_state[d_class, sum_class[d_class]-1] = old_state
		
		sum_nodes = np.append(0, sum_class.cumsum())
		prob_sum = np.array([self.tree.final_states_prob[sum_nodes[i]:sum_nodes[i+1]].sum() for i in range(len(sum_nodes)-1)])
		
		for period in range(nperiods):
			for k in range(self.dnum):
				d_sum = np.zeros(nperiods)
				old_state = 0
				for d_class in range(nperiods):
					for i in range(sum_class[d_class]):
						d_sum[d_class] += self.tree.final_states_prob[old_state] * self.d[k, old_state, period]
						old_state += 1
				for d_class in range(nperiods):
					for i in range(sum_class[d_class]):
						self.d[k, new_state[d_class, i], period] = d_sum[d_class] / prob_sum[d_class]

		old_state = 0
		for d_class in range(nperiods):
			for i in range(sum_class[d_class]):
				self.tree.final_states_prob[new_state[d_class, i]] = temp_prob[old_state]

		self.tree.node_prob[-len(self.tree.final_states_prob):] = self.tree.final_states_prob
		for p in range(1,nperiods-1):
			nodes = self.tree.get_nodes_in_period(p)
			for node in range(nodes[0], nodes[1]+1):
				worst_end_state, best_end_state = self.tree.reachable_end_states(node, period=p)
				self.tree.node_prob[node] = self.tree.final_states_prob[worst_end_state:best_end_state+1].sum()

	def _damage_interpolation(self):
		"""Create the interpolation coeffiecients used to calculate damages.
		"""
		if self.d is None:
			print("Importing stored damage simulation")
			self.import_damages()

		self._recombine_nodes()
		
		self.damage_coefs = np.zeros((self.tree.num_final_states, self.tree.num_periods, self.dnum-1, self.dnum))
		amat = np.ones((self.tree.num_periods, self.dnum, self.dnum))
		bmat = np.ones((self.tree.num_periods, self.dnum))

		self.damage_coefs[:, :, -1,  -1] = self.d[-1, :, :]
		self.damage_coefs[:, :, -1,  -2] = (self.d[-2, :, :] - self.d[-1, :, :]) / self.emit_pct[-2]
		amat[:, 0, 0] = 2.0 * self.emit_pct[-2]
		amat[:, 1:, 0] = self.emit_pct[:-1]**2
		amat[:, 1:, 1] = self.emit_pct[:-1]
		amat[:, 0, -1] = 0.0

		for state in range(0, self.tree.num_final_states):
			bmat[:, 0] = self.damage_coefs[state, :, -1,  -2] * self.emit_pct[-2]
			bmat[:, 1:] = self.d[:-1, state, :].T
			self.damage_coefs[state, :, 0] = np.linalg.solve(amat, bmat)

	def import_damages(self, loc="data/simulated_damages.csv"):
		try:
			with open(loc, 'r') as f:
				d = np.loadtxt(f, delimiter=";", comments="#")
		except IOError as e:
			import sys
			print("Could not import simulated damages:\n\t{}".format(e))
			sys.exit(0)

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
		ds = DamageSimulation(tree=self.tree, ghg_levels=self.ghg_levels, peak_temp=peak_temp,
					disaster_tail=disaster_tail, tip_on=tip_on, temp_map=temp_map, 
					temp_dist_params=temp_dist_params, maxh=maxh, cons_growth=cons_growth)
		self.d = ds.simulate(draws)
		return self.d

	def _forcing_based_mitigation(self, forcing, period): 
		"""Calculation of mitigation based on forcing up to period.

		Args:
			forcing (float): Cumulative forcing up to node.
			period (int): Period of node.

		Returns:
			float: Mitigation.
		"""
		p = period - 1
		if forcing > self.cum_forcings[p][1]:
			weight_on_sim2 = (self.cum_forcings[p][2] - forcing) / (self.cum_forcings[p][2] - self.cum_forcings[p][1])
			weight_on_sim3 = 0
		elif forcing > self.cum_forcings[p][0]:
			weight_on_sim2 = (forcing - self.cum_forcings[p][0]) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
			weight_on_sim3 = (self.cum_forcings[p][1] - forcing) / (self.cum_forcings[p][1] - self.cum_forcings[p][0])
		else:
			weight_on_sim2 = 0
			weight_on_sim3 = 1.0 + (self.cum_forcings[p][0] - forcing) / self.cum_forcings[p][0]
		
		return weight_on_sim2 * self.emit_pct[1] + weight_on_sim3*self.emit_pct[0]

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
		forcing = self.forcing.forcing_at_node(m, node)
		force_mitigation = self._forcing_based_mitigation(forcing, period)

		worst_end_state, best_end_state = self.tree.reachable_end_states(node, period=period)
		probs = self.tree.final_states_prob[worst_end_state:best_end_state+1]

		if force_mitigation < self.emit_pct[1]:
			damage = (probs *(self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 1] * force_mitigation \
					 + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 1, 2])).sum()
		
		elif force_mitigation < self.emit_pct[0]: #do dot product instead?
			damage = (probs * (self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 0] * force_mitigation**2 \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 1] * force_mitigation \
					  + self.damage_coefs[worst_end_state:best_end_state+1, period-1, 0, 2])).sum()
		
		######### what's happening here? ##############
		else: 
			damage = 0.0
			i = 0
			for state in range(worst_end_state, best_end_state+1): 
				if self.d[0, state, period-1] > 10e-4:
					deriv = 2.0 * self.damage_coefs[state, period-1, 0, 0]*self.emit_pct[0] \
							+ self.damage_coefs[state, period-1, 0, 1]
					decay_scale = deriv / (self.d[0, state, period-1]*np.log(0.5))
					dist = force_mitigation - self.emit_pct[0] + np.log(self.d[0, state, period-1]) \
						   / (np.log(0.5) * decay_scale) 
					damage += probs[i] * (0.5**(decay_scale*dist) * np.exp(-np.square(force_mitigation-self.emit_pct[0])/60.0))
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
	from optimization import GenericAlgorithm, GradientSearch
	from optimization import GAGradientSearch
	from output_functions import *

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

	m = np.array([ 0.65008568,  0.81440062,  0.57156974,  1.02442677,  0.94972038,
          0.80001195,  0.56711807,  1.25885777,  1.20418622,  1.22675892,
                 1.08200462,  1.059628  ,  0.97288246,  0.90817513,  0.41352478,
       1.2497254 ,  1.20470048,  1.19457448,  1.19942113,  1.25305931,
         1.14961153,  1.2680144 ,  1.26744892,  1.60484919,  1.54971032,
         1.56207038,  1.53777281,  1.62949171,  1.55358956,  0.97519823,
         0.92102587,  1.12843339,  1.37744208,  1.28497502,  1.0367739 ,
         1.26343189,  1.36535276,  1.0905257 ,  1.11133567,  1.28375479,
         1.19406048,  1.49001237,  1.08942412,  1.52660931,  1.30240349,
         1.30398985,  1.15276375,  1.23884217,  1.31855489,  1.23195591,
         1.30590214,  1.48104488,  1.48990133,  1.35075056,  1.39940133,
         1.31771028,  1.59167383,  1.62456475,  1.48355491,  1.66158315,
         1.49468617,  1.44681469,  1.2018028 ])
	"""
	m = np.array([0.63033831,0.79493609,0.63961916,1.00100798,0.89883409,0.90650158,0.63261796,
		1.27838304,1.22254166,1.26107556,0.93073007,1.29769024,0.93561291,0.95227753,0.56268567,
		1.29278316,1.15256622,1.2307686,1.11031776,1.23969716,1.11810618,1.55663593,1.42213896,
		1.25113711,1.12736059,1.60594014,1.48023671,1.71172097,1.59794234,1.33202771,0.42229699,
		2.02341063,1.61617494,1.76098407,1.45681979,1.80320135,1.4876755, 1.63924136,1.29798736,
		1.81450273,1.50371422,1.64708828,1.30479554,1.83540965,1.43450589,1.62121368,1.19193555,
		1.82614187,1.51256235,1.65722273,1.30874109,1.85953769,1.4531849, 1.64829521,1.20749883,
		1.94156574,1.53071302,1.70118658,1.24272015,3.35803589,2.59176097,0.91533447,0.0])
	"""

	u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0)
	#utility_t, cons_t, cost_t = u.utility(m, return_trees=True)
	#n_cons_t, cost_arr = delta_consumption(m, u, cons_t, cost_t, 0.01)
	#ssc_decomposition(m, u, utility_t, cons_t, cost_t, 0.01)
	

	ga_gs = GAGradientSearch(ga_pop=150, ga_generations=100, ga_cxprob=0.8, ga_mutprob=0.5, 
							upper_bound=3.0, gs_learning_rate=0.1, gs_iterations=100, 
							gs_acc=1e-07, num_features=63, utility=u)
	res = ga_gs.run()
	#print "Starting Generic Algorithm \n"
	#ga = GenericAlgorithm(300, 63, 50, 3.0, 0.80, 0.50, u.utility)
	#pop, fitness = ga.run()

	#sort_pop = pop[np.argsort(fitness)][::-1]
	
	#gs = GradientSearch(learning_rate=0.1, var_nums=63, utility=u, iterations=10, fixed_values=fixed_values)
	#res = gs.run(initial_point_list=[m], topk=1)
	

	#print "Moving over to Gradient Descent \n"
	#fixed_values = np.zeros(len(m))
	#fixed_values[0] = m[0]
	#m_hist = gd.run(u, m=m, alpha=0.1, num_iter=200)


