import numpy as np
import multiprocessing

class GenericAlgorithm(object):
	"""Optimization algorithm for the DLW model. 

	Args:
		pop_amount (int): Number of individuals in the population.
		num_feature (int): The number of elements in each individual, i.e. number of nodes in tree-model.
		num_generations (int): Number of generations of the populations to be evaluated.
		bound (float): Amount to reduce the 
		cx_prob (float): Probability of mating.
		mut_prob (float): Probability of mutation.
		utility (obj 'Utility'): Utility object containing the valuation function.
		constraints (ndarray): 1D-array of size (ind_size)

	TODO: Create and individual class.
	"""
	def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound, num_feature, utility):
		self.num_feature = num_feature
		self.pop_amount = pop_amount
		self.num_gen = num_generations
		self.cx_prob = cx_prob
		self.mut_prob = mut_prob
		self.u = utility
		self.bound = bound

	def _generate_population(self):
		"""Return 1D-array of random value in the given bound as the initial population.
	    
	    Returns:
	    	ndarray: Array of random value in the given bound with the shape of ('pop_amount', 'num_feature').
		"""
		#pop = np.random.random([self.pop_amount, self.num_feature]).cumsum(axis=1)*0.1
		pop = np.random.random([self.pop_amount, self.num_feature])*self.bound
		return pop

	def _evaluate(self, indvidual):
		"""Returns the utility of given individual.
	    
	    Parameters
	    	indvidual (ndarray or list): The shape of 'pop' define as 1 times of num_feature.
	   
	    Returns:
	    	ndarray: Array with utility at time zero.

		"""
		return self.u.utility(indvidual)

	def _select(self, pop, rate):
		"""Returns a 1D-array of selected individuals.
	    
	    Parameters:
		    pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
		    rate (float): The probability of an individual can be selected among population
		    
	    Returns:
	    	ndarray: Selected individuals.

		"""
		index = np.random.choice(self.pop_amount, int(rate*self.pop_amount), replace=False)
		return pop[index,:]

	def _random_index(self, individuals, size):
		"""Generate a random index of individuals of size 'size'.

		Args:
			individuals (ndarray or list): 2D-array of individuals.
			size (int): The number of indices to generate.
		
		Returns:
			ndarray: 1D-array of indices.

		"""
		inds_size = len(individuals)
		return np.random.choice(inds_size, size)

	def _selection_tournament(self, pop, k, tournsize, fitness):
	    """Select 'k' individuals from the input 'individuals' using 'k'
	    tournaments of 'tournsize' individuals.
	    
	    Args:
	    	individuals (ndarray or list): 2D-array of individuals to select from.
	    	k (int): The number of individuals to select.
	    	tournsize (int): The number of individuals participating in each tournament.
	   
	   	Returns:
	   		ndarray: Selected individuals.
	    
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
	    
	    Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').

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
	
	def _uniform_cross_over(self, pop, ind_prob):
		"""Performs a uniform cross-over of the population.
	    
	    Args:
	    	pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
	    	ind_prob (float): Probability of feature cross-over.
	    
		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			size = min(len(child1), len(child2))
			for i in range(size):
				if np.random.random() < ind_prob:
					child1[i], child2[i] = child2[i], child1[i]

	def _mutate(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability
		of 'self.mut_prob' of beeing selected and every element in this 
		individual has a probability 'ind_prob' of beeing mutated. The mutated
		value is a random number.

		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
	    	ind_prob (float): Probability of feature mutation.
	    	scale (float): The scaling of the random generated number for mutation. 

		"""
		pop_tmp = np.copy(pop)
		mutate_index = np.random.choice(self.pop_amount, int(self.mut_prob * self.pop_amount), replace=False)
		for i in mutate_index:
			feature_index = np.random.choice(self.num_feature, int(ind_prob * self.num_feature), replace=False)
			for j in feature_index:
				pop[i][j] = np.random.random()*scale
	
	def _uniform_mutation(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability
		of 'self.mut_prob' of beeing selected and every element in this 
		individual has a probability 'ind_prob' of beeing mutated. The mutated
		value is the current value plus a scaled uniform [-0.5,0.5] random value.

		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
	    	ind_prob (float): Probability of feature mutation.
	    	scale (float): The scaling of the random generated number for mutation.

	    """ 
		pop_len = len(pop)
		mutate_index = np.random.choice(pop_len, int(self.mut_prob * pop_len), replace=False)
		for i in mutate_index:
			prob = np.random.random(self.num_feature) 
			inc = (np.random.random(self.num_feature) - 0.5)*scale
			pop[i] += (prob > (1.0-ind_prob)).astype(int)*inc
			pop[i] = np.maximum(0.0, pop[i])

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
		"""

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
		The evolution steps:
			1. Select the individuals to perform cross-over and mutation.
			2. Cross over among the selected candidate.
			3. Mutate result as offspring.
			4. Combine the result of offspring and parent together. And selected the top 
			   80 percent of original population amount.
			5. Random Generate 20 percent of original population amount new individuals 
			   and combine the above new population.
		"""
		print("----------------Genetic Evolution Starting----------------")
		pop = self._generate_population()
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		fitness = pool.map(self._evaluate, pop) # how do we know pop[i] belongs to fitness[i]?
		fitness = np.array([val[0] for val in fitness])
		for g in range(0, self.num_gen):
			print ("-- Generation {} --".format(g+1))
			#pop_select = self._select(np.copy(pop), rate=1) # this works since we have rate=1 ?!
			pop_select = self._selection_tournament(pop, len(pop), 4, fitness)
			#self._two_point_cross_over(pop_select)
			self._uniform_cross_over(pop_select, 0.50)
			#self._mutate(pop_select, 0.10, np.exp(-g/self.num_gen))
			self._uniform_mutation(pop_select, 0.1, np.exp(-g/self.num_gen))

			fitness_select = pool.map(self._evaluate, pop_select)
			fitness_select = np.array([val[0] for val in fitness_select])
			
			pop_tmp = np.append(pop, pop_select, axis=0)
			fitness_tmp = np.append(fitness, fitness_select, axis=0)

			pop_survive, fitness_survive = self._survive(pop_tmp, fitness_tmp)

			pop_new = np.random.random([self.pop_amount - len(pop_survive), self.num_feature])*self.bound
			fitness_new = pool.map(self._evaluate, pop_new)
			fitness_new = np.array([val[0] for val in fitness_new])

			pop = np.append(pop_survive, pop_new, axis=0)
			fitness = np.append(fitness_survive, fitness_new, axis=0)
			self._show_evolution(fitness, pop)

		fitness = pool.map(self._evaluate, pop)
		fitness = np.array([val[0] for val in fitness])
		return pop, fitness



class GradientSearch(object) :
	"""
    reference the algorithm in http://cs231n.github.io/neural-networks-3/
	"""

	def __init__(self, learning_rate, var_nums, utility, accuracy=1e-06, iterations=100, 
				 step=0.00001, fixed_values=None):
		self.alpha = learning_rate
		self.u = utility
		self.var_nums = var_nums
		self.step = step
		self.accuracy = accuracy
		self.iterations = iterations
		self.fixed_values  = fixed_values
		if self.fixed_values is None:
			self.fixed_values = np.zeros(var_nums)
		self.non_zero_fv = np.where(self.fixed_values != 0.0)[0]

	def _initial_values(self, size):
		m = np.random.random(size) * 3.0
		return m
	
	def _gradient(self, x): # not used
		"""
		Use the centered formula for gradient calculation
		"""
		base = np.array([x] * len(x))
		shift = np.diag([self.step] * len(x))
		base_plus = base + shift
		base_minus = base - shift
		utility_plus = np.apply_along_axis(self.u.utility, 1, base_plus)
		utility_minus = np.apply_along_axis(self.u.utility, 1, base_minus)
		gradient_val = (utility_plus - utility_minus) / (2 * self.step)
		return gradient_val.flatten()
	
	def _dynamic_alpha(self, x_increase, grad_increase):
		if np.all(grad_increase == 0):
			return 0.0
		return np.abs(np.dot(x_increase, grad_increase) /  np.square(grad_increase).sum())


	def gradient_descent(self, initial_point):
		"""
		Annealing the learning rate. Step decay: Reduce the learning rate by some factor every few epochs.
		Typical values might be reducing the learning rate by a half every 5 epochs,
		"""
		learning_rate = self.alpha	
		num_decision_nodes = initial_point.shape[0]
		x_hist = np.zeros((self.iterations+1, num_decision_nodes))
		u_hist = np.zeros(self.iterations+1)
		u_hist[0] = self.u.utility(initial_point)
		x_hist[0] = initial_point
		prev_grad = 0.0
		half_iter = int(self.iterations / 2)

		for i in range(self.iterations):
			grad = self.u.numerical_gradient(x_hist[i])
			if i != 0:
				learning_rate = self._dynamic_alpha(x_hist[i]-x_hist[i-1], grad-prev_grad)

			new_x = x_hist[i] + grad*learning_rate
			new_x[self.non_zero_fv] = self.fixed_values[self.non_zero_fv]
			current = self.u.utility(new_x)[0]
			x_hist[i+1] = new_x
			u_hist[i+1] = current
			prev_grad = grad.copy()
			if i > half_iter:
				x_diff = np.abs(x_hist[i+1] - x_hist[i]).sum()
				u_diff = np.abs(u_hist[i+1] - u_hist[i])
				if x_diff < 1e-04 or u_diff < self.accuracy:
					print("Broke iteration..")
					break
			print("-- Interation {} -- \n Current Utility: {}".format(i+1, current))

		return x_hist[i+1], current

	def run(self, topk=4, initial_point_list=None, size=None):
		"""Initiate the gradient search algorithm. 

		Args:
			m (ndarray or list): 1D numpy array of size (num_decision_nodes).
			alpha (float): Step size in gradient descent.
			num_iter (int): Number of iterations to run.

		Returns:
			ndarray: The history of parameter vector, 2D numpy array of size (num_iter+1, num_decision_nodes) 

		"""
		print("----------------Gradient Search Starting----------------")
		if initial_point_list is None:
			if size is None:
				raise ValueError("Need size of the initial point array")
			initial_point_list = np.array(self._initial_values(size))

		if topk > len(initial_point_list):
			raise ValueError("topk {} > number of initial points {}".format(topk, len(initial_point_list)))

		candidate_points = initial_point_list[:topk]
		result = []
		count = 1
		for cp in candidate_points:
			if not isinstance(cp, np.ndarray):
				cp = np.array(cp)
			print cp
			print("Starting process {} of Gradient Descent".format(count))
			result.append(self.gradient_descent(cp))
			count += 1
		return result

class GAGradientSearch(object):
	def __init__(self, ga_pop, ga_generations, ga_cxprob, ga_mutprob, upper_bound, 
				 gs_learning_rate, gs_iterations, gs_acc, num_features, utility):
		self.ga_model = GenericAlgorithm(ga_pop, ga_generations, ga_cxprob, ga_mutprob, upper_bound, 
										 num_features, utility)
		self.gs_model = GradientSearch(gs_learning_rate, num_features, utility, gs_acc, 
									   gs_iterations)
	def run(self):
		final_pop, fitness = self.ga_model.run()
		sort_pop = final_pop[np.argsort(fitness)][::-1]
		res = self.gs_model.run(initial_point_list=sort_pop, topk=2)
		return res


