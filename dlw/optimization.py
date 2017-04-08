from __future__ import division, print_function
import numpy as np
import multiprocessing
from tools import _pickle_method, _unpickle_method
try:
    import copy_reg
except:
    import copyreg as copy_reg
import types

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class GenericAlgorithm(object):
	"""Optimization algorithm for the DLW-model. 

	Parameters
	----------
	pop_amount : int
		number of individuals in the population
	num_feature : int 
		number of elements in each individual, i.e. number of nodes in tree-model
	num_generations : int 
		number of generations of the populations to be evaluated
	bound : float
		upper bound of mitigation in each node
	cx_prob : float
		 probability of mating
	mut_prob : float
		probability of mutation.
	utility : `Utility` object
		object of utility class
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed

	Attributes
	----------
	pop_amount : int
		number of individuals in the population
	num_feature : int 
		number of elements in each individual, i.e. number of nodes in tree-model
	num_generations : int 
		number of generations of the populations to be evaluated
	bound : float
		upper bound of mitigation in each node
	cx_prob : float
		 probability of mating
	mut_prob : float
		probability of mutation.
	u : `Utility` object
		object of utility class
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed

	"""
	def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound, num_feature, utility,
				 fixed_values=None, fixed_indicies=None, print_progress=False):
		self.num_feature = num_feature
		self.pop_amount = pop_amount
		self.num_gen = num_generations
		self.cx_prob = cx_prob
		self.mut_prob = mut_prob
		self.u = utility
		self.bound = bound
		self.fixed_values = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress

	def _generate_population(self, size):
		"""Return 1D-array of random values in the given bound as the initial population."""
		pop = np.random.random([size, self.num_feature])*self.bound
		if self.fixed_values is not None:
			for ind in pop:
				ind[self.fixed_indicies] = self.fixed_values
		return pop

	def _evaluate(self, indvidual):
		"""Returns the utility of given individual."""
		return self.u.utility(indvidual)

	def _select(self, pop, rate):
		"""Returns a 1D-array of selected individuals.
	    
	    Parameters
	    ----------
	    pop : ndarray 
	    	population given by 2D-array with shape ('pop_amount', 'num_feature')
	    rate : float 
	    	the probability of an individual being selected
		    
	    Returns
	    -------
	    ndarray 
	    	selected individuals

		"""
		index = np.random.choice(self.pop_amount, int(rate*self.pop_amount), replace=False)
		return pop[index,:]

	def _random_index(self, individuals, size):
		"""Generate a random index of individuals of size 'size'.

		Parameters
		----------
		individuals : ndarray or list
			2D-array of individuals
		size : int 
			number of indices to generate
		
		Returns
		-------
		ndarray 
			1D-array of indices

		"""
		inds_size = len(individuals)
		return np.random.choice(inds_size, size)

	def _selection_tournament(self, pop, k, tournsize, fitness):
	    """Select `k` individuals from the input `individuals` using `k`
	    tournaments of `tournsize` individuals.
	    
	    Parameters
	    ----------
	    individuals : ndarray or list
	    	2D-array of individuals to select from
	    k : int
	    	 number of individuals to select
	    tournsize : int
	    	number of individuals participating in each tournament
	   
	   	Returns
	   	-------
	   	ndarray s
	   		selected individuals
	    
	    """
	    chosen = []
	    for i in xrange(k):
	        index = self._random_index(pop, tournsize)
	        aspirants = pop[index]
	        aspirants_fitness = fitness[index]
	        chosen_index = np.where(aspirants_fitness == np.max(aspirants_fitness))[0]
	        if len(chosen_index) != 0:
	        	chosen_index = chosen_index[0]
	        chosen.append(aspirants[chosen_index])
	    return np.array(chosen)

	def _two_point_cross_over(self, pop):
		"""Performs a two-point cross-over of the population.
	    
	    Parameters
	    ----------
		pop : ndarray 
			population given by 2D-array with shape ('pop_amount', 'num_feature')

		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			if np.random.random() <= self.cx_prob:
				cxpoint1 = np.random.randint(1, self.num_feature)
				cxpoint2 = np.random.randint(1, self.num_feature - 1)
				if cxpoint2 >= cxpoint1:
					cxpoint2 += 1
				else: # Swap the two cx points
					cxpoint1, cxpoint2 = cxpoint2, cxpoint1
				child1[cxpoint1:cxpoint2], child2[cxpoint1:cxpoint2] \
				= child2[cxpoint1:cxpoint2].copy(), child1[cxpoint1:cxpoint2].copy()
				if self.fixed_values is not None:
					child1[self.fixed_indicies] = self.fixed_values
					child2[self.fixed_indicies] = self.fixed_values
	
	def _uniform_cross_over(self, pop, ind_prob):
		"""Performs a uniform cross-over of the population.
	    
	    Parameters
	    ----------
	    pop : ndarray
	    	population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float
	    	probability of feature cross-over
	    
		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			size = min(len(child1), len(child2))
			for i in range(size):
				if np.random.random() < ind_prob:
					child1[i], child2[i] = child2[i], child1[i]

	def _mutate(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability of `mut_prob` of 
		beeing selected and every element in this individual has a probability `ind_prob` of beeing 
		mutated. The mutated value is a random number.

		Parameters
		----------
		pop : ndarray
			population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float 
	    	probability of feature mutation 
	    scale : float 
	    	scaling constant of the random generated number for mutation

		"""
		pop_tmp = np.copy(pop)
		mutate_index = np.random.choice(self.pop_amount, int(self.mut_prob * self.pop_amount), replace=False)
		for i in mutate_index:
			feature_index = np.random.choice(self.num_feature, int(ind_prob * self.num_feature), replace=False)
			for j in feature_index:
				if self.fixed_indicies is not None and j in self.fixed_indicies:
					continue
				else:
					pop[i][j] = max(0.0, pop[i][j]+(np.random.random()-0.5)*scale)
	
	def _uniform_mutation(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability of `mut_prob` of
		beeing selected and every element in this individual has a probability `ind_prob` of beeing 
		mutated. The mutated value is the current value plus a scaled uniform [-0.5,0.5] random value.

		Parameters
		----------
		pop : ndarray
			population given by 2D-array with shape ('pop_amount', 'num_feature')
	    ind_prob : float 
	    	probability of feature mutation 
	    scale : float 
	    	scaling constant of the random generated number for mutation

	    """ 
		pop_len = len(pop)
		mutate_index = np.random.choice(pop_len, int(self.mut_prob * pop_len), replace=False)
		for i in mutate_index:
			prob = np.random.random(self.num_feature) 
			inc = (np.random.random(self.num_feature) - 0.5)*scale
			pop[i] += (prob > (1.0-ind_prob)).astype(int)*inc
			pop[i] = np.maximum(0.0, pop[i])
			if self.fixed_values is not None:
				pop[i][self.fixed_indicies] = self.fixed_values

	def _show_evolution(self, fits, pop):
		"""Print statistics of the evolution of the population."""
		length = len(pop)
		mean = fits.mean()
		std = fits.std()
		min_val = fits.min()
		max_val = fits.max()
		print (" Min {} \n Max {} \n Avg {}".format(min_val, max_val, mean))
		print (" Std {} \n Population Size {}".format(std, length))
		print (" Best Individual: ", pop[np.argmax(fits)])

	def _survive(self, pop_tmp, fitness_tmp):
		"""The 80 percent of the individuals with best fitness survives to
		the next generation.

		Parameters
		----------
		pop_tmp : ndarray
			population
		fitness_tmp : ndarray
			fitness values of `pop_temp`

		Returns
		-------
		ndarray 
			individuals that survived

		"""
		index_fits  = np.argsort(fitness_tmp)[::-1]
		fitness = fitness_tmp[index_fits]
		pop = pop_tmp[index_fits]
		num_survive = int(0.8*self.pop_amount) 
		survive_pop = np.copy(pop[:num_survive])
		survive_fitness = np.copy(fitness[:num_survive])
		return np.copy(survive_pop), np.copy(survive_fitness)

	def run(self):
		"""Start the evolution process.
		
		The evolution steps are:
			1. Select the individuals to perform cross-over and mutation.
			2. Cross over among the selected candidate.
			3. Mutate result as offspring.
			4. Combine the result of offspring and parent together. And selected the top 
			   80 percent of original population amount.
			5. Random Generate 20 percent of original population amount new individuals 
			   and combine the above new population.

		Returns
		-------
		tuple
			final population and the fitness for the final population
		"""
		print("----------------Genetic Evolution Starting----------------")
		pop = self._generate_population(self.pop_amount)
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		fitness = pool.map(self._evaluate, pop) # how do we know pop[i] belongs to fitness[i]?
		fitness = np.array([val[0] for val in fitness])
		for g in range(0, self.num_gen):
			print ("-- Generation {} --".format(g+1))
			pop_select = self._select(np.copy(pop), rate=1)
			#pop_select = self._selection_tournament(pop, len(pop), 4, fitness)
			#self._two_point_cross_over(pop_select)
			self._uniform_cross_over(pop_select, 0.50)
			#do the check for mutation other here and save the indices
			self._uniform_mutation(pop_select, 0.20, np.exp(-float(g)/self.num_gen)**2)
			self._mutate(pop_select, 0.05)
			
			fitness_select = pool.map(self._evaluate, pop_select)
			fitness_select = np.array([val[0] for val in fitness_select])
			
			pop_tmp = np.append(pop, pop_select, axis=0)
			fitness_tmp = np.append(fitness, fitness_select, axis=0)

			pop_survive, fitness_survive = self._survive(pop_tmp, fitness_tmp)

			#pop_new = np.random.random([self.pop_amount - len(pop_survive), self.num_feature])*self.bound
			pop_new = self._generate_population(self.pop_amount - len(pop_survive))
			fitness_new = pool.map(self._evaluate, pop_new)
			fitness_new = np.array([val[0] for val in fitness_new])

			pop = np.append(pop_survive, pop_new, axis=0)
			fitness = np.append(fitness_survive, fitness_new, axis=0)
			if self.print_progress:
				self._show_evolution(fitness, pop)

		fitness = pool.map(self._evaluate, pop)
		fitness = np.array([val[0] for val in fitness])
		return pop, fitness



class GradientSearch(object) :
	"""Gradient search optimization algorithm for the DLW-model.

	Parameters
	----------
	utility : `Utility` object
		object of utility class
	learning_rate : float
		starting learning rate of gradient descent
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the gradient descent
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed
	scale_alpha : ndarray, optional
		array to scale the learning rate

	Attributes
	----------
	utility : `Utility` object
		object of utility class
	learning_rate : float
		starting learning rate of gradient descent
	var_nums : int
		number of elements in array to optimize
	accuracy : float
		stop value for the gradient descent
	fixed_values : ndarray, optional
		nodes to keep fixed
	fixed_indicies : ndarray, optional
		indicies of nodes to keep fixed
	print_progress : bool, optional
		if the progress of the evolution should be printed
	scale_alpha : ndarray, optional
		array to scale the learning rate

	"""

	def __init__(self, utility, learning_rate, var_nums, accuracy=1e-06, iterations=100, 
				 fixed_values=None, fixed_indicies=None, print_progress=False, scale_alpha=None):
		self.alpha = learning_rate
		self.u = utility
		self.var_nums = var_nums
		self.accuracy = accuracy
		self.iterations = iterations
		self.fixed_values  = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress
		self.scale_alpha = scale_alpha
		if scale_alpha is None:
			self.scale_alpha = np.exp(np.linspace(0.0, 3.0, var_nums))

	def _partial_grad(self, i):
		"""Calculate the ith element of the gradient vector."""
		m_copy = self.m.copy()
		m_copy[i] -= self.delta
		minus_utility = self.u.utility(m_copy)
		m_copy[i] += 2*self.delta
		plus_utility = self.u.utility(m_copy)
		grad = (plus_utility-minus_utility) / (2*self.delta)
		return grad, i

	def numerical_gradient(self, m, delta=1e-08, fixed_indicies=None):
		"""Calculate utility gradient numerically.

		Parameters
		----------
		m : ndarray or list
			array of mitigation
		delta : float, optional
			change in mitigation 
		fixed_indicies : ndarray or list, optional
			indicies of gradient that should not be calculated

		Returns
		-------
		ndarray
			gradient

		"""
		self.delta = delta
		self.m = m
		if fixed_indicies is None:
			fixed_indicies = []
		grad = np.zeros(len(m))
		if not isinstance(m, np.ndarray):
			self.m = np.array(m)
		pool = multiprocessing.Pool()
		indicies = np.delete(range(len(m)), fixed_indicies)
		res = pool.map(self._partial_grad, indicies)
		for g, i in res:
			grad[i] = g
		pool.close()
		pool.join()
		del self.m
		del self.delta
		return grad
	
	def _dynamic_alpha(self, x_increase, grad_increase, grad_size):
		"""Dynamic learning rate."""
		if np.all(grad_increase == 0):
			return np.zeros(grad_size)
		cons = np.abs(np.dot(x_increase, grad_increase) /  np.square(grad_increase).sum())
		return cons*self.scale_alpha

	def _ada_grad(self, cum_grad):
		epsilon = 1e-8
		ita = 0.001
		return 1/np.sqrt(cum_grad + epsilon) * ita

	def _rms_prop_revise(self, history_grad, grad_t, accumlate_dgrad):
		beta1 = 0.9
		beta2 = 0.999
		ita = 0.001
		eps = 1e-8
		def accelerate_factor(k, accumlate_dgrad):
			print("accumlate_dgrad: ", accumlate_dgrad)
			return (k / (1 + np.exp(np.abs(accumlate_dgrad))) - k / 2)

		acc_factor = accelerate_factor(5, accumlate_dgrad)
		E_g2 = np.mean(np.power(history_grad, 2), axis =0)
		E_g2 = 0.9 * E_g2 + 0.1 * np.power(grad_t, 2)
		return ita / np.sqrt(E_g2 + eps) * grad_t * (1)

	def _accelerate_scale(self, accelerator, history_grad, grad):
		sign_vector = np.sign(history_grad[-1] * grad)
		scale_vector = np.ones(self.var_nums) * ( 1 + 0.01)
		scale_vector[sign_vector < 0] = ( 1 - 0.01)
		return accelerator * scale_vector

	def _rms_prop(self, history_grad, grad_t):
		"""
		RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton in Lecture 6e
		of his Coursera Class.
		http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
		RMSprop and Adadelta have both been developed independently around the same time stemming
		from the need to resolve Adagrad's radically diminishing learning rates.
		"""

		beta1 = 0.9
		beta2 = 0.999
		ita = 0.001
		eps = 1e-8
		E_g2 = np.mean(np.power(history_grad, 2), axis =0)
		E_g2 = 0.9 * E_g2 + 0.1 * np.power(grad_t, 2)
		return ita / np.sqrt(E_g2 + eps) * grad_t

	def _adam(self, t, history_grad, grad_t):
		"""
		http://sebastianruder.com/optimizing-gradient-descent/index.html#fnref:15
		Adaptive Moment Estimation (Adam) [15] is another method that computes adaptive learning rates for each parameter.
		In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and
		RMSprop, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum
		"""
		beta1 = 0.9
		beta2 = 0.9
		ita = 0.002
		eps = 1e-8
		m_t = np.mean(history_grad, axis = 0)
		v_t = np.mean(np.power(history_grad, 2), axis =0)
		m_t = beta1 * m_t + (1 - beta1) * grad_t
		v_t = beta2 * v_t + (1 - beta2) * np.power(grad_t, 2)
		m_t = m_t / (1 - beta1**t)
		v_t = v_t / (1 - beta2**t)
		return ita * m_t / (np.sqrt(v_t) + eps)

	def gradient_descent(self, initial_point, return_last=False):
		"""Gradient descent algorithm. The `initial_point` is updated in the direction
		where the utility function is increases fastest.

		Parameters
		----------
		initial_point : ndarray
			initial guess of the mitigation
		return_last : bool, optional
			if True the function returns the last point, else the point 
				with highest utility

		Returns
		-------
		tuple
			(best point, best utility)
		
		"""
		learning_rate = self.alpha
		num_decision_nodes = initial_point.shape[0]
		x_hist = np.zeros((self.iterations+1, num_decision_nodes))
		u_hist = np.zeros(self.iterations+1)
		u_hist[0] = self.u.utility(initial_point)
		x_hist[0] = initial_point
		prev_grad = 0.0

		cum_grad = np.array(np.zeros(self.var_nums))
		history_grad = np.array(np.zeros(self.var_nums))
		accumlate_dgrad = np.array(np.zeros(self.var_nums))
		adam_rate = np.array(np.zeros(self.var_nums))
		rms_prop_rate = np.array(np.zeros(self.var_nums))
		accelerator = np.ones(self.var_nums)

		for i in range(self.iterations):
			grad = self.u.numerical_gradient(x_hist[i], fixed_indicies=self.fixed_indicies)
			grad[np.abs(grad) < 1e-9] = -1e-9
			cum_grad += np.power(grad,2)
			if i != 0:
				adam_rate = self._adam(i+1, history_grad, grad)
				accelerator = self._accelerate_scale(accelerator, history_grad, grad)
	
			new_x = x_hist[i] + adam_rate * accelerator
			new_x[new_x < 0] = 1e-8
			#new_x = x_hist[i] + rms_prop_rate
			history_grad = np.vstack([history_grad, grad])
			print("grade: ", grad)
			if self.fixed_values is not None:
				new_x[self.fixed_indicies] = self.fixed_values

			current = self.u.utility(new_x)[0]
			x_hist[i+1] = new_x
			u_hist[i+1] = current
			prev_grad = grad.copy()
	
			if self.print_progress:
				print("-- Interation {} -- \n Current Utility: {}".format(i+1, current))
				print(new_x)

		if return_last:
			return x_hist[i+1], u_hist[i+1]
		best_index = np.argmax(u_hist)
		return x_hist[best_index], u_hist[best_inde
		return x_hist[best_index], u_hist[best_index]

	def run(self, initial_point_list, topk=4):
		"""Initiate the gradient search algorithm. 

		Parameters
		----------
		initial_point_list : list
			list of initial points to select from
		topk : int, optional
			select and run gradient descent on the `topk` first points of 
			`initial_point_list`

		Returns
		-------
		tuple
			best mitigation point and the utility of the best mitigation point

		Raises
		------
		ValueError
			If `topk` is larger than the length of `initial_point_list`.

		"""
		print("----------------Gradient Search Starting----------------")
		
		if topk > len(initial_point_list):
			raise ValueError("topk {} > number of initial points {}".format(topk, len(initial_point_list)))

		candidate_points = initial_point_list[:topk]
		mitigations = []
		utilities = np.zeros(topk)
		for cp, count in zip(candidate_points, range(topk)):
			if not isinstance(cp, np.ndarray):
				cp = np.array(cp)
			print("Starting process {} of Gradient Descent".format(count+1))
			m, u = self.gradient_descent(cp)
			mitigations.append(m)
			utilities[count] = u
		best_index = np.argmax(utilities)
	
		return mitigations[best_index], utilities[best_index]


class NodeMaximum(object):
	"""Optimization node wise, keeping the rest of the nodes fixed. Starting from the 
	back of the tree and working back to node 0.
	"""

	@staticmethod
	def _min_func(x, m, i, utility):
  		m_copy = m.copy()
   		m_copy[i] = x
   		return -utility.utility(m_copy)[0]

	@classmethod
	def run(cls, m, utility):
		from scipy.optimize import fmin
		print("Utility before {}".format(utility.utility(m)[0]))
		print("Starting maximizing node wise..")
		for i in range(len(m))[::-1]:
			m[i] = fmin(cls._min_func, x0=m[i], args=(m, i, utility), disp=False)
		print("Done!")
		print("Utility after {}".format(utility.utility(m)[0]))
		return m








