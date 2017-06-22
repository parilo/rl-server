import pydot
import cv2
import numpy as np

# https://github.com/erocarrera/pydot/blob/master/pydot.py

class AssociationsGraph ():

    def __init__ (self, train_loop, som_state, som_action):

        self.train_loop = train_loop
        self.som_state = som_state
        self.som_action = som_action

        slen = self.som_state.get_clusters_count ()
        alen = self.som_action.get_clusters_count ()
        self.state_nodes = [None] * slen
        self.action_nodes = [None] * alen
        self.edges_s_to_a = [[None for x in range(alen)] for y in range(slen)]
        self.edges_a_to_s = [[None for x in range(slen)] for y in range(alen)]
        self.edges_s_to_s = {}

        self.graph = pydot.Dot(graph_type='digraph')

    def process_train_outputs (self):

        # rewards = self.train_outputs [0]
        actions = self.train_loop.train_outputs [1]
        prev_states = self.train_loop.train_outputs [2]
        next_states = self.train_loop.train_outputs [3]

        prev_states_clusters = self.som_state.get_clusters (prev_states)
        next_states_clusters = self.som_state.get_clusters (next_states)
        actions_clusters = self.som_action.get_clusters (actions)

        for ps, ns, a in zip (prev_states_clusters, next_states_clusters, actions_clusters):

            if (ps == ns):
                continue

            prev_state_node = self.state_nodes [ps]
            if (prev_state_node is None):
                prev_state_node = pydot.Node('s ' + str(ps), style="filled", fillcolor="green")
                self.state_nodes [ps] = prev_state_node
                self.graph.add_node(prev_state_node)

            # action_node = self.action_nodes [a]
            # if (action_node is None):
            #     action_node = pydot.Node('a ' + str(a), style="filled", fillcolor="pink")
            #     self.action_nodes [a] = action_node
            #     self.graph.add_node(action_node)

            next_state_node = self.state_nodes [ns]
            if (next_state_node is None):
                next_state_node = pydot.Node('s ' + str(ns), style="filled", fillcolor="green")
                self.state_nodes [ns] = next_state_node
                self.graph.add_node(next_state_node)

            # returns list of edges
            edge_key = 'e_' + str(ps) + '_' + str(ns)
            if (edge_key in self.edges_s_to_s):
                pass
            else:
                edge = pydot.Edge(prev_state_node, next_state_node)
                self.edges_s_to_s [edge_key] = edge
                self.graph.add_edge(edge)

        self.show_graph ()

    def show_graph(self):
        png = self.graph.create (format='png')
        file_bytes = np.asarray(bytearray(png), dtype=np.uint8)
        img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
        cv2.imshow ('Associations', img)
        cv2.waitKey (1)
