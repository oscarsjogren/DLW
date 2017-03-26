import numpy as np
from storage_tree import BigStorageTree

def gs_optimization(m, utility, fixed_values=None, fixed_indicies=None):
	""" the specialized optimization function, using both the generic algorithm and gradient descent"""
	from optimization import GradientSearch

	gs = GradientSearch(learning_rate=0.1, var_nums=len(m), utility=utility, 
						fixed_values=fixed_values, fixed_indicies=fixed_indicies, iterations=200)
	new_m, new_utility = gs.gradient_descent(m)
	return new_m

def gags_optimization(utility, ga_pop=150, ga_generations=300, ga_cxprob=0.8, ga_mutprob=0.50, 
					 upper_bound=3.0, gs_learning_rate=1.0, gs_iterations=50, gs_acc=1e-07, 
					 num_features=63, topk=3, fixed_values=None, fixed_indicies=None):
	
	from optimization import GenericAlgorithm, GradientSearch
	ga_model = GenericAlgorithm(pop_amount=ga_pop, num_generations=ga_generations, cx_prob=ga_cxprob, 
								mut_prob=ga_mutprob, bound=upper_bound, num_feature=num_features, 
								utility=utility, fixed_values=fixed_values, fixed_indicies=fixed_indicies)

	gs_model = GradientSearch(learning_rate=gs_learning_rate, var_nums=num_features, utility=utility, 
							  accuracy=gs_acc, iterations=gs_iterations, fixed_values=fixed_values,
							  fixed_indicies=fixed_indicies)
	
	final_pop, fitness = ga_model.run()
	sort_pop = final_pop[np.argsort(fitness)][::-1]
	res = gs_model.run(initial_point_list=sort_pop, topk=topk)
	return res


def additional_ghg_emission(m, utility):
	additional_emission = np.zeros(len(m))
	for i in range(len(m)):
		period = utility.tree.get_period(i)
		additional_emission[i] = (1.0 - m[i]) * utility.damage.bau.emission_to_ghg[period]
	return additional_emission

def ghg_level(utility, additional_emissions):
	""" dlw_tree_model in the end"""
	ghg_levels = np.zeros(len(additional_emissions))
	ghg_levels[0] = utility.damage.bau.ghg_start
	for i in range(1, len(ghg_levels)):
		ghg_levels[i] = ghg_levels[i-1] + additional_emissions[i-1]
	return ghg_levels

def store_price(m, utility, file_name, run_name, delimiter=';'):
	from tools import create_file
	d = create_file(file_name) # creates a file if does not already exists and returns the path
	price = utility.cost.price(0, m[0], 0)
	with open(d, 'a') as f:
		f.write(run_name + delimiter + str(price) + "\n")

def store_trees(prefix=None, start_year=2015, **kwargs):
	if prefix is None:
		prefix = ""
	for name, tree in kwargs.items():
		tree.write_columns(prefix + "trees", name, start_year)

def delta_consumption(m, utility, cons_tree, cost_tree, delta_m):
	m_copy = m.copy()
	m_copy[0] += delta_m
	fixed_values = np.array([m_copy[0]])
	fixed_indicies = np.array([0])
	new_m = gs_optimization(m_copy, utility, fixed_values, fixed_indicies)
	new_utility_tree, new_cons_tree, new_cost_tree, new_ce_tree = utility.utility(new_m, return_trees=True)

	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals()
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m
	
	return new_cons_tree, cost_array

def constraint_first_period(m, utility, first_node):
	fixed_values = np.repeat(first_node, 3)
	fixed_indicies = np.array([0,1,2])
	new_m, new_utility = gags_optimization(utility=utility, ga_generations=100, gs_iterations=5, fixed_values=fixed_values, fixed_indicies=fixed_indicies,
							  topk=2)
	new_utility_tree, new_cons_tree, new_cost_tree, new_ce_tree = utility.utility(new_m, return_trees=True)
	
	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals()
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m
	
	return new_m, new_cons_tree, cost_array

def find_ir(m, utility, payment, a=0.0, b=1.0): 
    """
      Function called by a zero root finder which is used
      to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
      the purpose of this function is to find the interest rate embedded in the EZ Utility model

      first calculate the utility with a final payment
    """
    from scipy.optimize import brentq

    def min_func(price):
    	utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
    	first_period_eps = payment * price
    	utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
    	return utility_with_final_payment - utility_with_initial_payment

    return brentq(min_func, a, b)

def find_term_structure(m, utility, num_periods, payment, a=0.0, b=0.99): # or find_ir
    """
      Function called by a zero root finder which is used
      to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
      the purpose of this function is to find the interest rate embedded in the EZ Utility model

      first calculate the utility with a final payment
    """
    from scipy.optimize import brentq

    def min_func(price):
    	period_cons_eps = np.zeros(num_periods)
    	period_cons_eps[-2] = payment
    	utility_with_payment = utility.adjusted_utility(m, period_cons_eps=period_cons_eps)

    	first_period_eps = payment * price
    	utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
    	return  utility_with_payment - utility_with_initial_payment

    return brentq(min_func, a, b)

def find_bec(m, utility, constraint_cost, a=-0.1, b=1.0):
    """Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

    Args:
    	m (ndarray): 1D-array of mitigation decisions.
    	utility (obj 'Utility'): Object where the utility is calculated.
    	constraint_cost (float): Difference in utility between two solutions.

    Returns:
    	float: Consumption of new solution.
    """
    from scipy.optimize import brentq

    def min_func(delta_con):
    	base_utility = utility.adjusted_utility(m)
    	new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
    	return new_utility - base_utility - constraint_cost

    return brentq(min_func, a, b)

def perpetuity_yield(price, start_date, a=0.1, b=10.0):
    """Function used to find the yield of a perpetuity starting at year 'start_date'.

    Args:
    	price (float): Price of zero coupon bond maturing at 'start_date'.
    	start_date (int): Start date of perpetuity.
    	a (float): Initial guess.
    	b (float): Initial guess.

    Returns:
    	float: Perpetuity yield.

    """
    from scipy.optimize import brentq

    def min_func(perp_yield):
    	return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

    return brentq(min_func, a, b)


def save_output(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, delta_cons_analysis=True,
				constraint_first_period=True, directory=None, prefix=None):
	from tools import write_columns_csv, append_to_existing
	import os
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
	ghg_levels = ghg_level(utility, additional_emissions)

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

	if directory is not None:
		start_filename = directory + os.path.sep
	else:
		start_filename = ""
	if prefix is not None:
		prefix += "_" 
	else:
		prefix = ""

	write_columns_csv([prices, m, ave_mitigations, ave_emissions, ghg_levels], start_filename + prefix + "node_period_output",
					   ["Node", "Mitigation", "Prices", "Average Mitigation", "Average Emission", "GHG Level"], [range(len(m))])

	append_to_existing([expected_period_price, expected_period_mitigation, expected_period_emissions],
						start_filename + prefix +  "node_period_output", header=["Period", "Expected Price", "Expected Mitigation",
						"Expected Emission"], index=[range(periods)])

	store_trees(prefix=start_filename+prefix, Utility=utility_tree, Consumption=cons_tree, 
				Cost=cost_tree, CertainEquivalence=ce_tree)

	
def save_sensitivity_analysis(m, utility, utility_tree, cons_tree, cost_tree, ce_tree, new_cons_tree, 
						 cost_array, start_filename):
	""" create_output in dlw_optimization. Maybe we only want to use gradient desecent here"""
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

	end_price = find_term_structure(m, utility, len(utility_tree), 0.01)
	perp_yield = perpetuity_yield(end_price, sdf_tree.periods[-2])
	print("Zero coupon bond maturing in {} has price {} and yield {}".format(sdf_tree.periods[-2], end_price, perp_yield))

	#grad = utility.numerical_gradient(m)
	#years_to_maturity = utility_tree.last_period - utility_tree.subinterval_len
	mu_0, mu_1, mu_2 = utility.marginal_utility(m, utility_tree, cons_tree, cost_tree, ce_tree)
	sub_len = sdf_tree.subinterval_len
	i = 1
	for period in sdf_tree.periods[1:]:
		node_period = sdf_tree.decision_interval(period)
		period_probs = utility.tree.get_probs_in_period(node_period)
		expected_damage = np.dot(new_cons_tree[period], period_probs)
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
		cross_sdf_damages[i] = np.dot(period_sdf, new_cons_tree[period]*period_probs)
		cov_term[i] = cross_sdf_damages[i] - expected_sdf[i]*expected_damage

		discount_prices[i] = expected_sdf[i]
		sdf_tree.set_value(period, period_sdf)

		if i < len(cost_array):
			net_discount_damages[i] = -(expected_damage + cost_array[i, 1]) * expected_sdf[i] / new_cons_tree.tree[0]
			cost_sum += -cost_array[i, 1] * expected_sdf[i] / new_cons_tree.tree[0]
		else:
			net_discount_damages[i] = -expected_damage * expected_sdf[i] / new_cons_tree.tree[0]

		risk_premiums[i] = -cov_term[i]/new_cons_tree.tree[0]
		discounted_expected_damages[i] = -expected_damage * expected_sdf[i] / new_cons_tree.tree[0]
		i += 1

	damage_scale = utility.cost.price(0, m[0], 0) / (net_discount_damages.sum()+risk_premiums.sum())
	scaled_discounted_ed = net_discount_damages * damage_scale
	scaled_risk_premiums = risk_premiums * damage_scale

	write_columns_csv([discount_prices, net_discount_damages, expected_damages, risk_premiums, expected_sdf, cross_sdf_damages, 
					   discounted_expected_damages, cov_term, scaled_discounted_ed, scaled_risk_premiums], start_filename + "sensitivity_output",
					   ["Year", "Discount Prices", "Net Expected Damages", "Expected Damages", "Risk Premium",
					    "Expected SDF", "Cross SDF & Damages", "Discounted Expected Damages", "Cov Term", "Scaled Net Expected Damages",
					    "Scaled Risk Premiums"], [sdf_tree.periods.astype(int)+2015]) 
	
	store_trees(prefix=start_filename, SDF=sdf_tree, DeltaConsumption=new_cons_tree, 
			    MU_0=mu_0, MU_1=mu_1, MU_2=mu_2)

