import numpy as np
import multiprocessing
from deap import base
from deap import creator
from deap import tools

class GenericAlgorithm(object):
	"""Optimization algorithm for the DLW model. 

	Args:
		num_pop (int): Number of individuals in the population.
		ind_size (int): The number of elements in each individual, i.e. number of nodes in tree-model.
		num_generations (int): Number of generations of the populations to be evaluated.
		cx_prob (float): Probability of mating.
		mut_prob (float): Probability of mutation.
		utility (obj 'Utility'): Utility object containing the valuation function.
		constraints (ndarray): 1D-array of size (ind_size)

	Parameters:
		num_pop (int): Number of individuals in the population.
		ind_size (int): The number of elements in each individual, i.e. number of nodes in tree-model.
		num_generations (int): Number of generations of the populations to be evaluated.
		cx_prob (float): Probability of mating.
		mut_prob (float): Probability of mutation.
		utility (obj 'Utility'): Utility object containing the valuation function.
		constraints (ndarray): 1D-array of size (ind_size)
		toolbox (obj 'deap.toolbox'): Deap object used in the run method.

	"""

	def __init__(self, num_pop, ind_size, num_generations, cx_prob, mut_prob, utility, 
				 fixed_values=None, start_values=None):
		self.num_pop = num_pop
		self.ind_size = ind_size
		self.num_gen = num_generations
		self.cx_prob = cx_prob
		self.mut_prob = mut_prob
		self.pop = None
		self.u = utility
		self.fixed_values = fixed_values
		if self.fixed_values is None:
			self.fixed_values = np.zeros(ind_size)
		self.non_zero_fv = np.where(self.fixed_values != 0.0)[0]
		self.start_values = start_values
		self._init_deap()

	def _attr_gen(self, individual, scaler):
		#m = np.random.random(self.ind_size).cumsum() * 0.1
		#m = np.ones(self.ind_size) + 2*(np.random.random(self.ind_size) - 0.5)
		#m[m<0.0] = 0.0
		if self.start_values is not None:
			ind = individual(self.start_values)
			return ind
		m = np.random.random(self.ind_size) * scaler
		m[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
		ind = individual(m)
		return ind

	def _average_twopoint(self, ind1, ind2):
		cxpoint1 = np.random.randint(1, self.ind_size)
		cxpoint2 = np.random.randint(1, self.ind_size - 1)
		if cxpoint2 >= cxpoint1:
			cxpoint2 += 1
		else: # Swap the two cx point
			cxpoint1, cxpoint2 = cxpoint2, cxpoint1
		ind2_copy = ind2.copy()
		ind2_copy[cxpoint1:cxpoint2] = ind1[cxpoint1:cxpoint2].copy()
		ind1[cxpoint1:cxpoint2] = (ind2[cxpoint1:cxpoint2] + ind1[cxpoint1:cxpoint2]) * 0.5
		ind1[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
		ind2_copy[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
		return ind1, ind2_copy

	def _cx_twopoint_copy(self, ind1, ind2):
	    """Execute a two points crossover with copy on the input individuals. The
	    copy is required because the slicing in numpy returns a view of the data,
	    which leads to a self overwritting in the swap operation.
	    
	    """
	    cxpoint1 = np.random.randint(1, self.ind_size)
	    cxpoint2 = np.random.randint(1, self.ind_size - 1)
	    if cxpoint2 >= cxpoint1:
	        cxpoint2 += 1
	    else: # Swap the two cx points
	        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

	    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
	        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
	    ind1[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
	    ind2[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
	    return ind1, ind2

	def _mutate_uniform(self, ind, indpb, step):
		prob = np.random.random(self.ind_size) 
		inc = (np.random.random(self.ind_size) - 0.5)
		ind += (prob > (1.0-indpb)).astype(int)*inc
		ind = np.maximum(0.0, ind)
		ind[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
		return ind

	def _one_point_mutation(self, ind, step):
		index = np.random.randint(self.ind_size)
		inc = (np.random.random() - 0.5)*step
		ind[index] += inc if (ind[index] + inc) > 0.0  else 0.0
		return ind


	def _init_deap(self):
		creator.create("FittnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FittnessMax)
		
		self.pool = multiprocessing.Pool()
		self.toolbox = base.Toolbox()
		self.toolbox.register("individual", self._attr_gen, creator.Individual, 3.0)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register("map", self.pool.map)
		self.toolbox.register("evaluate", self.u.utility)
		self.toolbox.register("mate", self._average_twopoint)
		self.toolbox.register("mutate", self._mutate_uniform, indpb=0.05)
		self.toolbox.register("select", tools.selTournament, tournsize=2)
		
	def run(self, num_reduce, print_output=True):
		pop = self.toolbox.population(n=self.num_pop)
		for g in range(0, self.num_gen):
			print "-- Generation {}--".format(g+1)

			fitnesses = list(self.toolbox.map(self.toolbox.evaluate, pop))
			for ind, fit in zip(pop, fitnesses):
				ind.fitness.values = fit
			
			fits = np.array([ind.fitness.values[0] for ind in pop])
			fits = fits[fits != 0.0]
			if len(fits) == 0:
				print "re-running, all zero fitness"
				self._init_deap()
				return self.run(num_reduce, print_output)

			length = len(pop)
			if print_output:
				mean = fits.mean()
				std = fits.std()
				min_val = fits.min()
				max_val = fits.max()
				print " Min {} \n Max {} \n Avg {}".format(min_val, max_val, mean)
				print " Std {} \n Population Size {}".format(std, length)
				print pop[np.argmax(fits)]
			
			new_size = int(0.99*length)
			size = new_size if new_size >= 100 else 100
			offspring = self.toolbox.select(pop, size)
			offspring = list(self.toolbox.map(self.toolbox.clone, offspring))
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if np.random.random() < self.cx_prob:
					self.toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:
				if np.random.random() < self.mut_prob:
					self.toolbox.mutate(mutant, step=np.exp(-g/float(self.num_gen)))
					del mutant.fitness.values

			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			pop[:] = offspring

		fits = np.array([ind.fitness.values[0] for ind in pop])
		return pop[np.argmax(fits)]



class GradientDescent(object):

	@classmethod
	def run(gd, utility, m=None, size=None, fixed_values=None, alpha=0.1, num_iter=100):
		"""
		Args:
			m (ndarray or list): 1D numpy array of size (num_decision_nodes).
			alpha (float): Step size in gradient descent.
			num_iter (int): Number of iterations to run.

		Returns:
			ndarray: The history of parameter vector, 2D numpy array of size (num_iter+1, num_decision_nodes) 

		"""
		if m is None:
			if size is None:
				raise ValueError("Need size of the mitigation array.")
			m = gd.initial_values(size)

		if fixed_values is None:
			fixed_values = np.zeros(len(m))
		non_zero_fv = np.where(fixed_values != 0.0)[0]
		num_decision_nodes = m.shape[0]
		m_hist = np.zeros((num_iter+1, num_decision_nodes))
		grad_hist = np.zeros((num_iter, num_decision_nodes))  #Initialize theta_hist
		m_hist[0] = m

		for i in range(num_iter):
			print "-- Iteration {} --".format(i+1)
			grad = utility.parallelized_num_gradient(m)
			grad_hist[i] = grad
			if i != 0:
				alpha = gd.dynamic_alpha(m_hist[i-1:i+1], grad_hist[i-1:i+1])
			if alpha == 0:
				break
			m += grad*alpha
			m[non_zero_fv] = fixed_values[non_zero_fv]
			m_hist[i+1] = m
			print utility.utility(m)
			print m
	
		return m_hist
	
	@classmethod	
	def initial_values(gd, size):
		m = np.random.random(size) * 3.0
		return m

	@classmethod
	def dynamic_alpha(gd, m_hist, grad_hist):
		grad_increase = grad_hist[-1]-grad_hist[-2]
		if np.all(grad_increase == 0):
			return 0.0
		return np.abs(np.dot(m_hist[-1]-m_hist[-2], grad_increase) /  np.square(grad_increase).sum())

