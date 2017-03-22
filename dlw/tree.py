import numpy as np

class TreeModel(object):
    """Tree model for the DLW-model. It provides the structure of a non-recombining tree used.

    Parameters:
        decision_times (ndarray or list): The years in the future where decisions will be made.
        prob_scale (float): Scaling constant for probabilities.

    """
    def __init__(self, decision_times, prob_scale):
        self.decision_times = decision_times
        if isinstance(self.decision_times, list):
            self.decision_times = np.array(self.decision_times)
        self.prob_scale = prob_scale
        self.num_periods = len(decision_times) - 1
        self.num_decision_nodes = 2**self.num_periods - 1
        self.num_final_states = 2**(self.num_periods - 1)
        self.final_states_prob = np.zeros(self.num_final_states)
        self.node_prob = np.zeros(self.num_decision_nodes)

        self._create_probs()

    def _create_probs(self):
        """Creates the probabilities of every nodes in the tree structure."""
        self.final_states_prob[0] = 1.0
        sum_probs = 1.0
        next_prob = 1.0

        ##Calculate the probability for the final state
        for n in range(1, self.num_final_states):
            next_prob = next_prob * self.prob_scale**(1.0 / n)
            self.final_states_prob[n] = next_prob
        self.final_states_prob /= np.sum(self.final_states_prob)

        self.node_prob[self.num_final_states-1:] = self.final_states_prob
        for period in range(self.num_periods-2, -1, -1):
            for state in range(0, 2**period):
                pos = self.get_node(period, state)
                self.node_prob[pos] = self.node_prob[2 * pos + 1] + self.node_prob[2 * pos + 2]

    def get_num_nodes_period(self, period):
        """Returns the number of nodes in the period."""
        if period >= self.num_periods:
            return 2**(self.num_periods-1)
        return 2**period

    def get_nodes_in_period(self, period):
        """Returns the specific nodes in the period."""
        if period >= self.num_periods:
            period = self.num_periods-1
        nodes = self.get_num_nodes_period(period)
        first_node = self.get_node(period, 0)
        return (first_node, first_node+nodes-1)

    def get_node(self, period, state):
        """Returns the node in period and state provided."""
        if state >= 2**period:
            raise ValueError("No such state in period {}".format(period))
        return 2**period + state - 1

    def get_state(self, node, period=None):
        """Returns the state the node represents."""
        if node >= self.num_decision_nodes:
            return node - self.num_decision_nodes
        if not period:
            period = self.get_period(node)
        return node - (2**period - 1)

    def get_period(self, node):
        """Returns what period the node is in."""
        if node >= self.num_decision_nodes: # can still be a too large node-number
            return self.num_periods

        for i in range(0, self.num_periods):
            if int((node+1) / 2**i ) == 1:
                return i

    def get_parent_node(self, child):
        """Returns the previous or parent node of the given child node."""
        if child == 0:
            return 0
        if child > self.num_decision_nodes:
            return child - self.num_final_states
        if child % 2 == 0:
            return int((child - 2) / 2)
        else:
            return int((child - 1 ) / 2)

    def get_path(self, node, period=None):
        """Returns the unique path taken to come to given node."""
        if period is None:
            period = self.tree.get_period(node)
        path = [node]
        for i in range(0, period):
            parent = self.get_parent_node(path[i])
            path.append(parent)
        path.reverse()
        return path

    def get_probs_in_period(self, period):
        """Returns the probabilities in given period."""
        first, last = self.get_nodes_in_period(period)
        return self.node_prob[range(first, last+1)]

    def reachable_end_states(self, node, period=None, state=None):
        """Returns what future end states can be reached from given node."""
        if period is None:
            period = self.get_period(node)
        if period >= self.num_periods:
            return (node - self.num_decision_nodes, node - self.num_decision_nodes)
        if state is None:
            state = self.get_state(node, period)

        k = self.num_final_states / 2**period
        k = int(k)
        return (k*state, k*(state+1)-1)
