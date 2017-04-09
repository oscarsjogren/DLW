============
Optimization
============

The :mod:`ez_climate.optimization` contains two classes, :class:`GeneticAlgorithm` and :class:`GradientSearch`. They both take an :class:`utility.Utility` object as an argument and tries to maximize the utility at time 0 by finding the optimal mitigation at every node of the tree. 


Genetic Algorithm (GA)
----------------------

The GA is an evolutionary algorithm, inspired by the evolution of species in nature. The evolution process starts from a population of vectors with uniformly distributed [0, :attr:`bound`] random elements. For each generation, the evolution steps in the :func:`run` method are:

  1. Select the individuals to perform cross-over and mutation.
  2. Cross over among the selected candidate.
  3. Mutate result as offspring.
  4. Combine the result of offspring and parent together. And selected the top 80 percent of original population amount.
  5. Random Generate 20 percent of original population amount new individuals and combine the above new population.

The mutation and cross-over methods are choosen to fit the optimization problem of the EZ-Climate model.

Gradient Search (GS)
--------------------

The GS uses the gradient descent algorithm and the numerical gradient to find the optimal mitigation points. Moveover, it uses the Adaptive Moment Estimation (Adam_) learning rate together with an accelarator scaler to update the points. Adam is a method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients, Adam also keeps an exponentially decaying average of past gradients. The accelerator is used to amplify low gradient values of mitigation values in nodes in the end of the tree, and thus reduce computation time. The :func:`run` method takes :attr:`initial_point_list` and :attr:`topk` as arguments, runs the gradient descent optimization of the :attr:`topk` first elements of the :attr:`initial_point_list`, and picks the resulting point with the highest utility. 


GA and GS together
------------------

An example of how to use the :class:`GeneticAlgorithm` and :class:`GradientSearch` can be found `here <../examples/output_paper.html>`_. The :func:`GradientSearch.run` takes the last generation population of the :func:`GeneticAlgorithm.run` as the :attr:`initial_point_list` argument and performs the gradient descent optimization with these as intial guess. 


.. _Adam: http://sebastianruder.com/optimizing-gradient-descent/index.html#fnref:15