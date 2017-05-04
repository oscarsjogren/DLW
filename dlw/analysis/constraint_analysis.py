from __future__ import division, print_function
from ez_climate.tools import write_columns_csv
import tools

class ConstraintAnalysis(object):

	def __init__(self, utility):
		self.utility = utility 
		self.delta_util_c = None
		self.delta_util_x = None
		self.delta_con = None
		self.marginal_benefit = None

	def save_constraint_analysis(self, cfp_m, utility, delta_util_x, delta_cons=0.01, prefix=None):
		self.delta_util_x = delta_util_x
		utility_given_delta_con = self.utility.adjusted_utility(cfp_m, first_period_consadj=delta_cons)
		
		self.delta_util_c = utility_given_delta_con - self.utility.utility(cfp_m)
		self.delta_con = tools.find_bec(cfp_m, self.utility, delta_util_x)
		self.marginal_benefit = (delta_util_x / self.delta_util_c) * self.delta_con * self.utility.cost.cons_per_ton / self.delta_cons
		self.delta_cons_billions = self.delta_con * self.utility.cost.cons_per_ton * self.utility.damage.bau.emit_level[0]
		delta_emission_gton = self.delta_cons * self.utility.damage.bau.emit_level[0]
		deadweight = self.delta_con * self.utility.cost.cons_per_ton / self.delta_cons
		

	def save_output(self, prefix=None):
		if prefix is not None:
			prefix += "_" 
		else:
			prefix = ""
		write_columns_csv([self.delta_util_x, self.delta_util_c, [self.delta_con], self.marginal_benefit, [self.delta_cons_billions],
						   [self.delta_emission_gton], [self.deadweight]], prefix+"constraint_output",
						   header=["Delta Utility Mitigation", "Delta Utility Consumption", "Delta Consumption", 
						   "Marginal Benefit", "Delta Consumption Billions", "Delta Emission GTon", "Deadweight"])