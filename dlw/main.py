
from tree import TreeModel
from bau import DLWBusinessAsUsual
from cost import DLWCost
from damage import DLWDamage
from utility import EZUtility
from analysis import *
from tools import *
from optimization import *

t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0)
bau_default_model = DLWBusinessAsUsual()
bau_default_model.bau_emissions_setup(t)
c = DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=2000.0, max_price=2500.0,
			tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0, max_penalty=0.01, penalty_scale=1.25)

df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000])
#df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
#		temp_map=1, temp_dist_params=None, maxh=100.0, cons_growth=0.015)
df.import_damages()
df.forcing_init(sink_start=35.596, forcing_start=4.926, ghg_start=400, partition_interval=5,
	forcing_p1=0.13173, forcing_p2=0.607773, forcing_p3=315.3785, absorbtion_p1=0.94835,
	absorbtion_p2=0.741547, lsc_p1=285.6268, lsc_p2=0.88414)

m = np.array([0.69033867,0.86901052,0.67112552,1.06174977,0.97328999,0.98026425
			,0.55130198,1.15782437,1.1470986, 1.17761229,1.04984526,1.21856301
			,0.96739168,0.7803391, 0.40705306,0.99897783,0.99742034,1.01264097
			,1.01051397,1.01683717,1.0147632, 1.16182373,1.13290088,1.020123
			,1.02182366,1.28713016,1.33503218,1.57391551,1.09963307,0.78621563
			,0.58371202,1.0006737, 1.00445704,1.01195281,1.04572151,0.9963772
			,0.9936301, 1.01214498,1.07212626,0.99500166,0.98708549,0.99819421
			,1.09999692,0.97000199,0.92702554,1.01677141,1.18936162,0.99819644
			,0.99816849,1.26935332,0.85649588,0.9744461, 0.96990773,0.24921964
			,0.73818256,1.06512262,1.89854024,1.41898561,1.56183604,1.41962237
			,0.69262425,1.4678787, 0.86732196])

u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0)

#utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
ga_model = GenericAlgorithm(pop_amount=200, num_generations=250, cx_prob=0.8, mut_prob=0.5, 
						bound=2.0, num_feature=63, utility=u, print_progress=True)
gs_model = GradientSearch(learning_rate=0.001, var_nums=63, utility=u, accuracy=1e-8, 
						  iterations=100, print_progress=True)
#final_pop, fitness = ga_model.run()
#sort_pop = final_pop[np.argsort(fitness)][::-1]

#m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
#m_opt, u_opt = gs_model.run(initial_point_list=[m], topk=1)
for i in range(63):
	plot_mitigation_at_node(m, i, u, save=True)
#m_opt = sort_pop[0]
#m_opt = m
m_opt = NodeMaximum.run(m_opt, u)


#utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
#save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t)
#save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t)

# Constraint first period mitigation to 0.0
#cfp_m, cfp_cons_tree, cfp_cost_array = constraint_first_period(m_opt, u, 0.0)
#cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
#save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t)
#save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, 
#							  cfp_cons_tree, cfp_cost_array, "CFP")

