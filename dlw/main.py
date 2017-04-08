
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
			tech_const=1.5, tech_scale=0.0, cons_at_0=30460.0)

df = DLWDamage(tree=t, bau=bau_default_model, cons_growth=0.015, ghg_levels=[450, 650, 1000], subinterval_len=5)
#df.damage_simulation(draws=4000000, peak_temp=6.0, disaster_tail=18.0, tip_on=True, 
#		temp_map=1, temp_dist_params=None, maxh=100.0, cons_growth=0.015)
df.import_damages()

m = np.array([ 0.67756584,0.85225621,0.67448498,1.06584066,0.96022359,0.94935034
,0.64422064,1.16076658,1.15793594,1.20075048,1.10341129,1.24891172
,1.03953605,0.9579818, 0.4377837, 1.00073521,1.00053618,1.00389282
,1.00392819,1.00159948,1.00158316,1.09816663,1.09774883,1.00148481
,1.00148403,1.21101867,1.18873258,1.40193278,1.36990709,0.89664789
,0.54428276,1.00019353,1.00002115,0.99983874,1.00004625,0.99926064
,0.99960826,0.9994324, 0.99947741,0.99956738,0.99995048,0.99989551
,0.99977021,0.9996163, 0.99976571,0.99978149,1.00000033,0.99923965
,0.9993017, 0.99958442,0.99918414,0.99883929,0.9989224, 1.07179224
,1.04130021,1.01060381,0.97572908,1.0692045, 1.08034373,2.00755976
,1.47565507,0.93021933,0.00961698])


u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, add_penalty_cost=True, max_penalty=0.001, penalty_scale=1.0)

utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
ga_model = GenericAlgorithm(pop_amount=200, num_generations=250, cx_prob=0.8, mut_prob=0.5, 
						bound=2.0, num_feature=63, utility=u, print_progress=True)
gs_model = GradientSearch(learning_rate=0.001, var_nums=63, utility=u, accuracy=1e-8, 
						  iterations=100, print_progress=True)
#final_pop, fitness = ga_model.run()
#sort_pop = final_pop[np.argsort(fitness)][::-1]

#m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
#m_opt, u_opt = gs_model.run(initial_point_list=[m], topk=1)

#m_opt = sort_pop[0]
#for i in range(63):
#	plot_mitigation_at_node(m, i, u, save=True, prefix="")

#m_opt = m
#m_opt = NodeMaximum.run(m_opt, u)	

#utility_t, cons_t, cost_t, ce_t = u.utility(m_opt, return_trees=True)
#save_output(m_opt, u, utility_t, cons_t, cost_t, ce_t)
#save_sensitivity_analysis(m_opt, u, utility_t, cons_t, cost_t, ce_t)

# Constraint first period mitigation to 0.0
#cfp_m = constraint_first_period(m_opt, u, 0.0)
#cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t = u.utility(cfp_m, return_trees=True)
#save_output(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, prefix="CFP")
#save_sensitivity_analysis(cfp_m, u, cfp_utility_t, cfp_cons_t, cfp_cost_t, cfp_ce_t, "CFP")

# everything else in run can easily be created too