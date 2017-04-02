from __future__ import division
import numpy as np


class Forcing(object):
	"""Forcing of GHG emissions for the DLW-model.

	what do write for these constants? Where do they come from?

	Parameters:
		tree (obj: 'TreeModel'): Provides the tree structure used.
		bau (obj: 'BusinessAsUsual'): Provides the business as usual case.
		sink_start (float): Sinking constant.
		forcing_start (float): 
		ghg_start (int): Today's GHG-level.
		partition_interval (int): The interval, in years, where forcing is calculated.
		forcing_p1 (float):
		forcing_p2 (float):
		forcing_p3 (float):
		absorbtion_p1 (float):
		absorbtion_p2 (float):
		lsc_p1 (float):
		lsc_p2 (float):

	"""
	def __init__(self, tree, bau, sink_start, forcing_start, ghg_start, partition_interval,
				 forcing_p1, forcing_p2, forcing_p3, absorbtion_p1, absorbtion_p2, lsc_p1, lsc_p2):
		self.tree = tree
		self.bau = bau
		self.sink_start = sink_start
		self.forcing_start = forcing_start
		self.ghg_start = ghg_start
		self.partition_interval = partition_interval
		self.forcing_p1 = forcing_p1
		self.forcing_p2 = forcing_p2
		self.forcing_p3 = forcing_p3
		self.absorbtion_p1 = absorbtion_p1
		self.absorbtion_p2 = absorbtion_p2
		self.lsc_p1 = lsc_p1
		self.lsc_p2 = lsc_p2

	def _forcing_and_ghg_at_node(self, m, node, k=None, returning="forcing"):
		"""Calculates the forcing based mitigation leading up to the damage calculation in "node".

		Args:
			m (ndarray): Array of mitigations in each node. 
			node (int): The node for which the forcing leading to the
				damages is being calculated.
			k (int, optional): The ghg-path in cum_forcings to update.

		Returns:
			float: foricing at node.

		"""
		if node == 0 and returning == "forcing":
			return 0.0
		elif node == 0 and returning== "ghg":
			return self.ghg_start

		period = self.tree.get_period(node)
		path = self.tree.get_path(node, period)

		period_lengths = self.tree.decision_times[1:period+1] - self.tree.decision_times[:period]
		increments = period_lengths/self.partition_interval

		cum_sink = self.sink_start
		cum_forcing = self.forcing_start
		ghg_level = self.ghg_start

		for p in range(0, period):
			start_emission = (1.0 - m[path[p]]) * self.bau.emission_by_decisions[p]
			if p < self.tree.num_periods-1: 
				end_emission = (1.0 - m[path[p]]) * self.bau.emission_by_decisions[p+1]
			else:
				end_emission = start_emission
			increment = int(increments[p])
			for i in range(0, increment):
				p_co2_emission = start_emission + i * (end_emission-start_emission) / increment
				p_co2 = 0.71 * p_co2_emission # where are these numbers coming from?
				p_c = p_co2 / 3.67 
				add_p_ppm = self.partition_interval * p_c / 2.13
				lsc = self.lsc_p1 + self.lsc_p2 * cum_sink
				absorbtion = 0.5 * self.absorbtion_p1 * np.sign(ghg_level-lsc) * np.abs(ghg_level-lsc)**self.absorbtion_p2
				cum_sink += absorbtion
				#cum_forcing += self.forcing_p1 * np.sign(ghg_level-self.forcing_p3)*np.abs(ghg_level-self.forcing_p3)**self.forcing_p2
				cum_forcing += self.forcing_p1*np.abs(ghg_level-self.forcing_p3)**self.forcing_p2
				ghg_level += add_p_ppm - absorbtion

		if returning == "forcing":
			return cum_forcing
		elif returning == "ghg":
			return ghg_level
		else:
			raise ValueError("Does not recognize the returning string {}".format(returning))
	

	def forcing_at_node(self, m, node, k=None):
		return self._forcing_and_ghg_at_node(m, node, k, returning="forcing")

	def ghg_level_at_node(self, m, node):
		return self._forcing_and_ghg_at_node(m, node, returning="ghg")
		
