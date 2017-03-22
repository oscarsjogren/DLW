import numpy as np
from scipy.optimize import brentq
from storage_tree import BigStorageTree

def optimization(m, utility, fixed_values):
	""" the specialized optimization function, using both the generic algorithm and gradient descent"""
	from optimization import GradientSearch

	gs = GradientSearch(learning_rate=0.1, var_nums=len(m), utility=utility,
						fixed_values=fixed_values, iterations=50)
	new_m, new_utility = gs.gradient_descent(m)
	return new_m

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


def save_output_result(m, utility, utility_tree, cons_tree, cost_tree, dir='data'):
	from tools import write_csv_1D
	bau = utility.damage.bau
	tree = utility.tree
	prices = np.zeros(len(m))
	ave_mitigations = np.zeros(len(m))
	ave_emissions = np.zeros(len(m))
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
	#save prices, ave_mitigation, ave_emissions, ghg_levels, utility, cons, cost
	write_csv_1D(prices, "prices")
	write_csv_1D(ave_mitigations, "ave_mitigation")
	write_csv_1D(ave_emissions, "ave_emissions")
	write_csv_1D(ghg_levels, "ghg_levels")
	utility_tree.write_csv("utility_tree")
	cons_tree.write_decision_times_csv("consumption_tree")
	cost_tree.write_decision_times_csv("cost_tree")


def delta_consumption(m, utility, cons_tree, cost_tree, delta_m):
	m_copy = m.copy()
	m_copy[0] += delta_m
	fixed_values = np.zeros(len(m_copy))
	fixed_values[0] = m_copy[0]
	new_m = optimization(m, utility, fixed_values)
	new_utility_tree, new_cons_tree, new_cost_tree = utility.utility(new_m, return_trees=True)

	for period in new_cons_tree.periods:
		new_cons_tree.tree[period] = (new_cons_tree.tree[period]-cons_tree.tree[period]) / delta_m

	first_period_intervals = new_cons_tree.first_period_intervals()
	cost_array = np.zeros((first_period_intervals, 2))
	for i in range(first_period_intervals):
		potential_consumption = (1.0 + utility.cons_growth)**(new_cons_tree.subinterval_len * i)
		cost_array[i, 0] = potential_consumption * cost_tree[0]
		cost_array[i, 1] = (potential_consumption * new_cost_tree[0] - cost_array[i, 0]) / delta_m

	return new_cons_tree, cost_array

def ssc_decomposition(m, utility, utility_tree, cons_tree, cost_tree, delta_m):
	""" create_output in dlw_optimization. Maybe we only want to use gradient desecent here"""

	sdf_tree = BigStorageTree(utility.period_len, utility.decision_times)
	sdf_tree.set_value(0, np.array([1.0]))

	discount_prices = np.zeros(len(sdf_tree))
	net_exptected_damages = np.zeros(len(sdf_tree))
	risk_premium = np.zeros(len(sdf_tree))
	cost_sum = 0
	discounted_expected_damages = 0
	net_discounted_expected_damages = 0
	risk_premium = 0
	delta_cons_tree, delta_cost_array = delta_consumption(m, utility, cons_tree, cost_tree, delta_m)

	years_to_maturity = utility_tree.last_period - utility_tree.subinterval_len
	num_periods = len(utility_tree)-1
	#discount_prices[-1] = find_term_structure(m, utility, len(utility_tree)-1, 0.01)
	discount_prices[-1] = 0.001
	#grad = utility.numerical_gradient(m)
	mu_0, mu_1, mu_2 = utility.marginal_utility(m, utility_tree, cons_tree, cost_tree)
	sub_len = sdf_tree.subinterval_len
	for period in sdf_tree.periods[1:]:
		node_period = sdf_tree.decision_interval(period)
		expected_damages = np.dot(delta_cons_tree.tree[period], utility.tree.get_probs_in_period(node_period))
		period_probs = utility.tree.get_probs_in_period(node_period)

		if sdf_tree.is_information_period(period-sdf_tree.subinterval_len):
			total_probs = period_probs[::2] + period_probs[1::2]
			sdf = (np.repeat(total_probs, 2) / period_probs) * np.repeat(mu_1.tree[period-sub_len]/mu_0.tree[period-sub_len], 2)
		else:
			sdf = mu_1.tree[period-sub_len]/mu_0.tree[period-sub_len]

		period_sdf = sdf_tree.tree[period-sub_len]*sdf
		expected_sdf = np.dot(period_sdf, period_probs)
		cross_sdf_damages = np.dot(period_sdf, delta_cons_tree.tree[period]*period_probs)
		cov_term = cross_sdf_damages - expected_sdf*expected_damages

		discount_prices[period] = expected_sdf
		sdf_tree.set_value(period, period_sdf)

		if not sdf_tree.is_decision_period(period):
			net_discount_damage = -(expected_damages + cost_array[period, 1]) * expected_sdf / delta_cons_tree.tree[period]
			cost_sum += - cost_array[period, 1] * expected_sdf / delta_cons_tree.tree[period]
		else:
			net_discount_damage = -expected_damages * expected_sdf / delta_cons_tree.tree[period]

		net_exptected_damages[period] = net_discount_damage
		risk_premium[period] = -cov_term/delta_cons_tree.tree[period]
		discounted_expected_damages += -expected_damages * expected_sdf / delta_cons_tree.tree[period]
		net_discounted_expected_damages += net_discount_damage
		risk_premium += risk_premium[period]

	total = risk_premium + net_discounted_expected_damages
	price = utility.cost.price(0, m[0], 0)
	ed = net_discounted_expected_damages/total * price
	rp = risk_premium/total * price
	print (price, ed, rp)


def find_ir(m, utility, payment, a=0.0, b=1.0):
    """
      Function called by a zero root finder which is used
      to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
      the purpose of this function is to find the interest rate embedded in the EZ Utility model

      first calculate the utility with a final payment
    """
    def min_func(price):
    	utility_with_final_payment = utility.adjusted_utility(m, final_cons_eps=payment)
    	first_period_eps = payment * price
    	utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
    	return utility_with_final_payment - utility_with_initial_payment

    return brentq(min_func, a, b)

def find_term_structure(m, utility, num_periods, payment, a=0.0, b=1.0): # or find_ir
    """
      Function called by a zero root finder which is used
      to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
      the purpose of this function is to find the interest rate embedded in the EZ Utility model

      first calculate the utility with a final payment
    """
    def min_func(price):
    	period_cons_eps = np.zeros(num_periods)
    	period_cons_eps[-1] = payment
    	utility_with_payment = utility.adjusted_utility(m, period_cons_eps=period_cons_eps)

    	first_period_eps = payment * price
    	utility_with_initial_payment = utility.adjusted_utility(m, first_period_consadj=first_period_eps)
    	return  utility_with_payment - utility_with_initial_payment

    return brentq(min_func, a, b)


'''
   function to call from optimizer to find the break-even consumption to equalize utility from a constrained optimization
'''
def find_bec(m, utility, constraint_cost, a=-0.1, b=1.0):
    """Used to find a value for consumption that equalizes utility at time 0 in two different solutions.

    Args:
    	m (ndarray): 1D-array of mitigation decisions.
    	utility (obj 'Utility'): Object where the utility is calculated.
    	constraint_cost (float): Difference in utility between two solutions.

    Returns:
    	float: Consumption of new solution.
    """
    def min_func(delta_con):
    	base_utility = utility.adjusted_utility(m)
    	new_utility = utility.adjusted_utility(m, first_period_consadj=delta_con)
    	return new_utility - base_utility - constraint_cost

    return brentq(min_func, a, b)

def perpetuity_yield(self, perp_yield, price, start_date):
    '''
      Function called by a zero root finder which is used
      to find the yield of a perpetuity starting at year start_date

    '''

    return price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield
