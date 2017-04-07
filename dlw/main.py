
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

m = np.array([ 0.67961073,0.85796937,0.67153906,1.06338793,0.96561344,0.95898243
,0.60506171,1.15836071,1.15703593,1.19308078,1.0858612, 1.23641389
,1.00225317,0.87719471,0.45776965,1.0036049, 1.00366799,1.0058377
,1.00568702,1.00710488,1.00642136,1.11569044,1.11343087,1.01035858
,1.01131088,1.24717774,1.28281215,1.49000946,1.25434465,0.91422178
,0.47031894,0.9996656, 1.00048618,1.00042412,0.99890683,1.00045561
,1.00020798,0.99694714,0.99903464,0.99853715,0.9987289, 0.9975014
,0.99840207,0.99810216,0.99754464,0.99927602,1.00143186,0.99664025
,0.99802374,0.99533977,0.9962839, 0.99968455,0.99891384,0.73253735
,0.94108126,1.0174935, 1.37967819,1.30926058,1.31087989,1.89947442
,0.82992153,1.20784532,0.72409751])


u = EZUtility(tree=t, damage=df, cost=c, period_len=5.0, add_penalty_cost=True, max_penalty=0.001, penalty_scale=1.0)

utility_t, cons_t, cost_t, ce_t = u.utility(m, return_trees=True)
ga_model = GenericAlgorithm(pop_amount=200, num_generations=250, cx_prob=0.8, mut_prob=0.5, 
						bound=2.0, num_feature=63, utility=u, print_progress=True)
gs_model = GradientSearch(learning_rate=0.001, var_nums=63, utility=u, accuracy=1e-8, 
						  iterations=100, print_progress=True)
final_pop, fitness = ga_model.run()
sort_pop = final_pop[np.argsort(fitness)][::-1]

#m_opt, u_opt = gs_model.run(initial_point_list=sort_pop, topk=1)
#m_opt, u_opt = gs_model.run(initial_point_list=[m], topk=1)

#m_opt = sort_pop[0]
m_opt = m
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