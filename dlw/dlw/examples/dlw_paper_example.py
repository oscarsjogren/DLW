from dlw import (TreeModel, DLWBusinessAsUsual, DLWCost, DLWDamage,
				EZUtility, GenericAlgorithm, GradientDescent as gd,
				find_ir)

t = TreeModel()
bau_default_model = DLWBusinessAsUsual()
bau_default_model.bau_emissions_setup(t)
c = DLWCost(t, bau_default_model.emit_level[0])
df = DLWDamage(tree=t, bau=bau_default_model)
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

u = EZUtility(tree=t, damage=df, cost=c, cons_growth=0.015, 
	period_len=5.0, decision_times=[0, 15, 45, 85, 185, 285, 385])
utility_t, cons_t, cost_t, ce_tree = u.utility(m, return_trees=True)


#print "Starting Generic Algorithm \n"
#ga = GenericAlgorithm(500, 63, 400, 0.80, 0.50, u)
#m = ga.run(0)



#print "Moving over to Gradient Descent \n"

#m_hist = gd.run(m, u, alpha=1e-6, num_iter=500)
