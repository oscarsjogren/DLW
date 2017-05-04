
import numpy as np
from tree import TreeModel
from bau import DLWBusinessAsUsual
from cost import DLWCost
from damage import DLWDamage
from utility import EZUtility
from analysis import RiskDecomposition, ClimateOutput, ConstraintAnalysis
from tools import *
from optimization import  GeneticAlgorithm, GradientSearch

m = np.array([ 0.74931503,0.91562481,0.70245901,1.11948303,0.97124621,0.96823361
,0.71502781,1.15358141,1.13174278,1.18244938,1.14936183,1.23352911
,0.88338659,0.73759418,0.55734468,1.01460519,0.99770894,1.02446396
,1.00552805,1.02910602,1.01462017,1.05811006,1.02222637,1.0377256
,1.02483132,1.35153686,0.95559224,1.26641559,1.50528127,0.921781
,0.68471511,0.97307307,0.96263887,0.97839762,0.97412634,0.97676557
,0.96713563,0.99025996,0.98486475,0.98100176,0.97061012,0.984765
,1.00417745,0.96963769,0.96710202,1.01287332,0.91422436,0.97448233
,0.96612273,0.97588121,0.97596596,1.02549839,1.05589004,1.30568944
,0.96821265,1.31291838,1.04916108,0.97320771,0.53598762,1.04753574
,1.16611029,0.57021441,0.32158966])

if __name__ == "__main__":
	header, indices, data = import_csv("DLW_research_runs", indices=2)
	for i in range(13, 21):
		name = indices[i][1]
		a, ra, eis, pref, temp, tail, growth, tech_chg, tech_scale, joinp, maxp, on, maps = data[i]
		print(name, ra, eis)
		if on == 1.0:
			on = True
		else:
			on = False
		maps = int(maps)
		t = TreeModel(decision_times=[0, 15, 45, 85, 185, 285, 385], prob_scale=1.0)
		bau_default_model = DLWBusinessAsUsual()
		bau_default_model.bau_emissions_setup(t)
		c = DLWCost(t, bau_default_model.emit_level[0], g=92.08, a=3.413, join_price=joinp, max_price=maxp,
					tech_const=tech_chg, tech_scale=tech_scale, cons_at_0=30460.0)

		df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=growth, ghg_levels=[450, 650, 1000], subinterval_len=5)
		#df.damage_simulation(draws=4000000, peak_temp=temp, disaster_tail=tail, tip_on=on, 
		#					 temp_map=maps, temp_dist_params=None, maxh=100.0, cons_growth=growth)
		df.import_damages()

		u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=eis, ra=ra, time_pref=pref)

		if a <= 2.0:
			ga_model = GeneticAlgorithm(pop_amount=150, num_generations=75, cx_prob=0.8, mut_prob=0.50, 
								bound=1.5, num_feature=63, utility=u, print_progress=True)
			
			gs_model = GradientSearch(var_nums=63, utility=u, accuracy=1e-8, 
							  iterations=250, print_progress=True)
	
			final_pop, fitness, u_hist = ga_model.run()
			sort_pop = final_pop[np.argsort(fitness)][::-1]
			
			m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
			#m_opt, u_opt = gs_model.run(initial_point_list=[m], topk=1)
			rd = RiskDecomposition(u)
			rd.sensitivity_analysis(m_opt)
			rd.save_output(m_opt, prefix=name)

			co = ClimateOutput(u)
			co.calculate_output(m_opt)
			co.save_output(m_opt, prefix=name)

			

		# Constraint first period mitigation to 0. NEEDS TO BE FIZED FOR NEW STRUCTURE
		else:
			cfp_m = constraint_first_period(u, 0.0, t.num_decision_nodes)
			cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(m_opt, return_trees=True)
			save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, prefix="CFP_"+name)
			delta_utility = save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t,
												    "CFP_"+name, return_delta_utility=True)
			delta_utility_x = delta_utility - cfp_utility_t[0]
			save_constraint_analysis(cfp_m, u, delta_utility_x, prefix="CFP_"+name)
"""
m_arr = []
names = []
files = ["test_0.2_1.0_2Eps-Zin(10)_node_period_output", "test_0.2_1.0_2base_case_node_period_output",
"test_0.2_1.0_2Eps-Zin(5)_node_period_output"]
for f in files:
	name = f[14:]
	m = import_csv(f, header=True, indices=1)[2][:, 0]
	m_arr.append(m)
	names.append(name)


u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, eis=eis, ra=ra, time_pref=pref,
					  add_penalty_cost=True, max_penalty=0.2, penalty_scale=1.0)
cd = CoordinateDescent(u, 63, iterations=1)
cd.run(m_arr[7])
ghg_level = df.ghg_level(m_arr[7], 6)
write_columns_to_existing([ghg_level], "test_ghg_levels_penalty", header=[names[7]])

"""
