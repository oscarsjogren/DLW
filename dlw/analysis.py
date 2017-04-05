from __future__ import division, print_function
import numpy as np
from storage_tree import BigStorageTree


def additional_ghg_emission(m, utility):
	"""Calculate the emission added by every node.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	
	Returns
	-------
	ndarray
		additional emission in nodes
	
	"""
	additional_emission = np.zeros(len(m))
	cache = set()
	for node in range(utility.tree.num_final_states, len(m)):
		path = utility.tree.get_path(node)
		for i in range(len(path)):
			if path[i] not in cache:
				additional_emission[path[i]] = (1.0 - m[path[i]]) *  utility.damage.bau.emission_to_ghg[i]
				cache.add(path[i])
	return additional_emission

def store_trees(prefix=None, start_year=2015, **kwargs):
	"""Saves values of `BaseStorageTree` objects. The file is saved into the 'data' directory
	in the current working directory. If there is no 'data' directory, one is created. 

	Parameters
	----------
	prefix : str, optional 
		prefix to be added to file_name
	start_year : int, optional
		start year of analysis
	**kwargs 
		arbitrary keyword arguments of `BaseStorageTree` objects

	"""
	if prefix is None:
		prefix = ""
	for name, tree in kwargs.items():
		tree.write_columns(prefix + "trees", name, start_year)

def delta_consumption(m, utility, cons_tree, cost_tree, delta_m):
	"""Calculate the changes in consumption and the mitigation cost component 
	of consumption when increaseing period 0 mitigiation with `delta_m`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	cons_tree : `BigStorageTree` object
		consumption storage tree of consumption values
		from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost storage tree of cost values from optimal mitigation values
	delta_m : float 
		value to increase period 0 mitigation by
	
	Returns
	-------
	tuple
		(storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	from optimization import GradientSearch

	m_copy = m.copy()
	m_copy[0] += delta_m
	fixed_values = np.array([m_copy[0]])
	fixed_indicies = np.array([0])
	gs = GradientSearch(learning_rate=0.0001, var_nums=len(m), utility=utility, fixed_values=fixed_values,
					    fixed_indicies=fixed_indicies, iterations=50, print_progress=True)
	#new_m, new_utility = gs.gradient_descent(m_copy)
	new_m = np.array([ 0.01, 0.78824846,0.58508793,1.0793555, 0.87276349,0.83227749
					,0.59321162,1.23932217,1.23715003,1.12651173,0.70515073,1.02599857
					,0.81972626,0.71812895,0.60490766,1.0000786, 0.99991533,0.99597102
					,1.00342248,1.24811976,1.22865303,1.26025693,1.25489448,1.3550156
					,1.39898312,1.63839418,1.39452366,1.6707743, 0.87281147,0.50159931
					,0.59023953,0.99964796,1.02726416,1.08777123,0.97123516,1.08684712
					,1.04915634,0.99814674,0.71325348,0.93816392,1.00285859,0.96058354
					,0.93463693,1.67369065,1.71097296,1.50254306,2.20806497,1.00503636
					,1.00486623,0.98147334,0.86463592,0.74216614,1.29771723,1.0178112
					,0.66534999,1.14538711,0.39910146,0.99323851,1.89178438,1.96957663
					,1.93852741,1.26593956,1.14926961])
	new_utility_tree, new_cons_tree, new_cost_tree, new_ce_tree = utility.utility(new_m, return_trees=True)

	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m
	
	return new_cons_tree, cost_array

def constraint_first_period(m, utility, first_node):
	"""Calculate the changes in consumption, the mitigation cost component of consumption,
	and new mitigation values when constraining the first period mitigation to `first_node`.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	first_node : float
		value to constrain first period to
	
	Returns
	-------
	tuple
		(new mitigation array, storage tree of changes in consumption, ndarray of costs in first sub periods)

	"""
	from optimization import GenericAlgorithm, GradientSearch
	fixed_values = np.array([first_node])
	fixed_indicies = np.array([0])
	ga_model = GenericAlgorithm(pop_amount=250, num_generations=200, cx_prob=0.8, mut_prob=0.5, bound=3.0,
								num_feature=len(m), utility=utility, fixed_values=fixed_values, 
								fixed_indicies=fixed_indicies, print_progress=True)

	gs_model = GradientSearch(learning_rate=0.01, var_nums=len(m), utility=utility, accuracy=1e-7,
							  iterations=100, fixed_values=fixed_values, fixed_indicies=fixed_indicies, 
							  print_progress=True)

	#final_pop, fitness = ga_model.run()
	#sort_pop = final_pop[np.argsort(fitness)][::-1]
	#new_m, new_utility = gs_model.run(initial_point_list=sort_pop, topk=1)
	new_m = np.array([ 0.,0.7882696, 0.58508105,1.07943186,0.8731261, 0.83241017
					,0.59330286,1.23992245,1.23717968,1.1270462, 0.7050969, 1.02625494
					,0.81955393,0.71807122,0.60503416,1.0005276, 1.0005276, 0.99536339
					,1.00412582,1.24954285,1.22904588,1.26034804,1.25495678,1.35538958
					,1.39976507,1.638699,1.39446657,1.67095233,0.87280804,0.50154055
					,0.5902515, 0.99931359,1.02783932,1.08821456,0.97095914,1.08737374
					,1.04944987,0.99851374,0.71316054,0.93899318,1.00331359,0.96113676
					,0.93450013,1.67405632,1.71108871,1.50264644,2.20814373,1.00546816
					,1.00594499,0.98282418,0.86421222,0.7407718, 1.29791955,1.01760887
					,0.66529758,1.14576276,0.39895113,0.99319427,1.89182733,1.9695831
					,1.93856945,1.26595294,1.14928655])
	return new_m


def find_ir(m, utility, payment, a=0.0, b=1.0): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	.. note:: requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(price):
		utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return utility_with_final_payment - utility_with_initial_payment

	return brentq(min_func, a, b)

def find_term_structure(m, utility, payment, a=0.0, b=0.99): 
	"""Find the price of a bond that creates equal utility at time 0 as adding `payment` to the value of 
	consumption in the final period. The purpose of this function is to find the interest rate 
	embedded in the `EZUtility` model. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	payment : float
		value added to consumption in the final period
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	.. note:: requires the 'scipy' package

	"""
	from scipy.optimize import brentq
	def min_func(price):
		period_cons_eps = np.zeros(int(utility.decision_times[-1]/utility.period_len) + 1)
		period_cons_eps[-2] = payment
		utility_with_payment = utility.adjusted_utility(m, period_cons_eps=period_cons_eps)

		first_period_eps = payment * price
		utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
		return  utility_with_payment - utility_with_initial_payment

	return brentq(min_func, a, b)

def find_bec(m, utility, constraint_cost, a=-0.1, b=1.0):
	"""Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	constraint_cost : float
		utility cost of constraining period 0 to zero
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	.. note:: requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(delta_con):
		base_utility = utility.adjusted_utility(m)
		new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
		return new_utility - base_utility - constraint_cost

	return brentq(min_func, a, b)

def perpetuity_yield(price, start_date, a=0.1, b=10.0):
	"""Find the yield of a perpetuity starting at year `start_date`.

	Parameters
	----------
	price : float
		price of bond ending at `start_date`
	start_date : int
		start year of perpetuity
	a : float, optional
		initial guess
	b : float, optional
		initial guess - f(b) needs to give different sign than f(a)
	
	Returns
	-------
	tuple
		result of optimization

	.. note:: requires the 'scipy' package

	"""
	from scipy.optimize import brentq

	def min_func(perp_yield):
		return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

	return brentq(min_func, a, b)


def save_output(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, prefix=None):
	"""Save the result of optimization and calculated values based on optimal mitigation. For every node the 
	function calculates and saves:
		* average mitigation
		* average emission
		* GHG level 
		* SCC 
	into the file `prefix` + 'node_period_output' in the 'data' directory in the current working directory. 

	For every period the function calculates and appends:
		* expected SCC/price
		* expected mitigation 
		* expected emission 
	into the file  `prefix` + 'node_period_output' in the 'data' directory in the current working directory. 

	The function also saves the values stored in the `BaseStorageTree` object parameters to a file called 
	`prefix` + 'tree' in the 'data' directory in the current working directory. If there is no 'data' 
	directory, one is created. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	utility_tree : `BigStorageTree` object
		utility values from optimal mitigation values
	cons_tree : `BigStorageTree` object
		consumption values from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost values from optimal mitigation values
	ce_tree : `BigStorageTree` object
		certain equivalence values from optimal mitigation values
	prefix : str, optional
		prefix to be added to file_name

	"""
	from tools import write_columns_csv, append_to_existing
	bau = utility.damage.bau
	tree = utility.tree
	periods = tree.num_periods
	prices = np.zeros(len(m))
	ave_mitigations = np.zeros(len(m))
	ave_emissions = np.zeros(len(m))
	expected_period_price = np.zeros(periods)
	expected_period_mitigation = np.zeros(periods)
	expected_period_emissions = np.zeros(periods)
	additional_emissions = additional_ghg_emission(m, utility)
	ghg_levels = utility.damage.ghg_level(m)

	periods = tree.num_periods
	for period in range(0, periods):
		years = tree.decision_times[period]
		nodes = tree.get_nodes_in_period(period)
		num_nodes_period = 1 + nodes[1] - nodes[0]
		period_lens = tree.decision_times[:period+1] 
		for node in range(nodes[0], nodes[1]+1):
			path = np.array(tree.get_path(node, period))
			new_m = m[path]
			mean_mitigation = np.dot(new_m, period_lens) / years
			price = utility.cost.price(years, m[node], mean_mitigation)
			prices[node] = price
			ave_mitigations[node] = utility.damage.average_mitigation_node(m, node, period)
			ave_emissions[node] = additional_emissions[node] / (num_nodes_period*bau.emission_to_bau)
		probs = tree.get_probs_in_period(period)
		expected_period_price[period] = np.dot(prices[nodes[0]:nodes[1]+1], probs)
		expected_period_mitigation[period] = np.dot(ave_mitigations[nodes[0]:nodes[1]+1], probs)
		expected_period_emissions[period] = np.dot(ave_emissions[nodes[0]:nodes[1]+1], probs)

	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""

	write_columns_csv([m, prices, ave_mitigations, ave_emissions, ghg_levels], prefix + "node_period_output",
					   ["Node", "Mitigation", "Prices", "Average Mitigation", "Average Emission", "GHG Level"], [range(len(m))])

	append_to_existing([expected_period_price, expected_period_mitigation, expected_period_emissions],
						prefix +  "node_period_output", header=["Period", "Expected Price", "Expected Mitigation",
						"Expected Emission"], index=[range(periods)], start_char='\n')

	store_trees(prefix=prefix, Utility=utility_tree, Consumption=cons_tree, 
				Cost=cost_tree, CertainEquivalence=ce_tree)

	
def save_sensitivity_analysis(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, prefix=None):
	"""Calculate and save sensitivity analysis based on the optimal mitigation. For every sub-period, i.e. the 
	periods given by the utility calculations, the function calculates and saves:
		* discount prices
		* net expected damages
		* expected damages
		* risk premium
		* expected SDF
		* cross SDF & damages
		* discounted expected damages
		* cov term
		* scaled net expected damages
		* scaled risk premiums
	into the file  `prefix` + 'sensitivity_output' in the 'data' directory in the current working directory. 

	Furthermore, for every node the function calculates and saves:
		* SDF 
		* delta consumption
		* forward marginal utility  
		* up-node marginal utility
		* down-node marginal utility
	into the file `prefix` + 'tree' in the 'data' directory in the current working directory. If there is no 'data' 
	directory, one is created. 

	Parameters
	----------
	m : ndarray or list
		array of mitigation
	utility : `Utility` object
		object of utility class
	utility_tree : `BigStorageTree` object
		utility values from optimal mitigation values
	cons_tree : `BigStorageTree` object
		consumption values from optimal mitigation values
	cost_tree : `SmallStorageTree` object
		cost values from optimal mitigation values
	ce_tree : `BigStorageTree` object
		certain equivalence values from optimal mitigation values
	prefix : str, optional
		prefix to be added to file_name

	"""
	from tools import write_columns_csv

	sdf_tree = BigStorageTree(utility.period_len, utility.decision_times)
	sdf_tree.set_value(0, np.array([1.0]))

	discount_prices = np.zeros(len(sdf_tree))
	net_expected_damages = np.zeros(len(sdf_tree))
	expected_damages = np.zeros(len(sdf_tree))
	risk_premiums = np.zeros(len(sdf_tree))
	expected_sdf = np.zeros(len(sdf_tree))
	cross_sdf_damages = np.zeros(len(sdf_tree))
	discounted_expected_damages = np.zeros(len(sdf_tree))
	net_discount_damages = np.zeros(len(sdf_tree))
	cov_term = np.zeros(len(sdf_tree))

	discount_prices[0] = 1.0
	cost_sum = 0

	end_price = find_term_structure(m, utility, 0.01)
	perp_yield = perpetuity_yield(end_price, sdf_tree.periods[-2])
	print("Zero coupon bond maturing in {} years has price {} and perpetuity yield {}".format(int(sdf_tree.periods[-2]), end_price, perp_yield))
	# save -^ somewhere

	delta_cons_tree, delta_cost_array = delta_consumption(m, utility, cons_tree, cost_tree, 0.01)
	mu_0, mu_1, mu_2 = utility.marginal_utility(m, utility_tree, cons_tree, cost_tree, ce_tree)
	sub_len = sdf_tree.subinterval_len
	i = 1
	for period in sdf_tree.periods[1:]:
		node_period = sdf_tree.decision_interval(period)
		period_probs = utility.tree.get_probs_in_period(node_period)
		expected_damage = np.dot(delta_cons_tree[period], period_probs)
		expected_damages[i] = expected_damage
		
		if sdf_tree.is_information_period(period-sdf_tree.subinterval_len):
			total_probs = period_probs[::2] + period_probs[1::2]
			mu_temp = np.zeros(2*len(mu_1[period-sub_len]))
			mu_temp[::2] = mu_1[period-sub_len]
			mu_temp[1::2] = mu_2[period-sub_len]
			sdf = (np.repeat(total_probs, 2) / period_probs) * (mu_temp/np.repeat(mu_0[period-sub_len], 2))
			period_sdf = np.repeat(sdf_tree.tree[period-sub_len],2)*sdf 
		else:
			sdf = mu_1[period-sub_len]/mu_0[period-sub_len]
			period_sdf = sdf_tree[period-sub_len]*sdf 

		expected_sdf[i] = np.dot(period_sdf, period_probs)
		cross_sdf_damages[i] = np.dot(period_sdf, delta_cons_tree[period]*period_probs)
		cov_term[i] = cross_sdf_damages[i] - expected_sdf[i]*expected_damage

		discount_prices[i] = expected_sdf[i]
		sdf_tree.set_value(period, period_sdf)

		if i < len(delta_cost_array):
			net_discount_damages[i] = -(expected_damage + delta_cost_array[i, 1]) * expected_sdf[i] / delta_cons_tree[0]
			cost_sum += -delta_cost_array[i, 1] * expected_sdf[i] / delta_cons_tree[0]
		else:
			net_discount_damages[i] = -expected_damage * expected_sdf[i] / delta_cons_tree[0]

		risk_premiums[i] = -cov_term[i]/delta_cons_tree[0]
		discounted_expected_damages[i] = -expected_damage * expected_sdf[i] / delta_cons_tree[0]
		i += 1

	damage_scale = utility.cost.price(0, m[0], 0) / (net_discount_damages.sum()+risk_premiums.sum())
	scaled_discounted_ed = net_discount_damages * damage_scale
	scaled_risk_premiums = risk_premiums * damage_scale

	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""

	write_columns_csv([discount_prices, net_discount_damages, expected_damages, risk_premiums, expected_sdf, cross_sdf_damages, 
					   discounted_expected_damages, cov_term, scaled_discounted_ed, scaled_risk_premiums], prefix + "sensitivity_output",
					   ["Year", "Discount Prices", "Net Expected Damages", "Expected Damages", "Risk Premium",
					    "Expected SDF", "Cross SDF & Damages", "Discounted Expected Damages", "Cov Term", "Scaled Net Expected Damages",
					    "Scaled Risk Premiums"], [sdf_tree.periods.astype(int)+2015]) 
	
	store_trees(prefix=prefix, SDF=sdf_tree, DeltaConsumption=delta_cons_tree, 
			    MU_0=mu_0, MU_1=mu_1, MU_2=mu_2)

