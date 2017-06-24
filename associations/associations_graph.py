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
        # alen = self.som_action.get_clusters_count ()
        self.state_nodes = [None] * slen
        # self.action_nodes = [None] * alen
        # self.edges_s_to_a = [[None for x in range(alen)] for y in range(slen)]
        # self.edges_a_to_s = [[None for x in range(slen)] for y in range(alen)]
        self.edges_s_to_s = {}

        self.graph = pydot.Dot(graph_type='digraph')
        self.graph_show_index = 0
        self.graph_show_every = 200

    def process_train_outputs (self):

        rewards = self.train_loop.train_outputs [0]
        actions = self.train_loop.train_outputs [1]
        prev_states = self.train_loop.train_outputs [2]
        next_states = self.train_loop.train_outputs [3]

        prev_states_clusters = self.som_state.get_clusters (prev_states)
        next_states_clusters = self.som_state.get_clusters (next_states)
        actions_clusters = self.som_action.get_clusters (actions)

        for ps, ns, a, r in zip (prev_states_clusters, next_states_clusters, actions_clusters, rewards):

            if (ps == ns):
                continue

            prev_state_node = None
            prev_state_node_info = self.state_nodes [ps]
            if (prev_state_node_info is None):
                prev_state_node = pydot.Node('s ' + str(ps), style="filled", fillcolor="green")
                prev_state_node_info = {
                    'id' : str(ps),
                    'node' : prev_state_node,
                    'count' : 1,
                    'added' : False,
                    'edges' : {},
                    'edges_count' : 0,
                    'action_counts' : {}
                }
                self.state_nodes [ps] = prev_state_node_info
            else:
                prev_state_node = prev_state_node_info ['node']
                prev_state_node_info ['count'] += 1

            action_key = str(a)
            if (action_key in prev_state_node_info ['action_counts']):
                prev_state_node_info ['action_counts'][action_key] += 1
            else:
                prev_state_node_info ['action_counts'][action_key] = 1


            next_state_node = None
            next_state_node_info = self.state_nodes [ns]
            if (next_state_node_info is None):
                next_state_node = pydot.Node('s ' + str(ns), style="filled", fillcolor="green")
                next_state_node_info = {
                    'id' : str(ns),
                    'node' : next_state_node,
                    'count' : 1,
                    'added' : False,
                    'edges' : {},
                    'edges_count' : 0,
                    'action_counts' : {}
                }
                self.state_nodes [ns] = next_state_node_info
            else:
                next_state_node = next_state_node_info ['node']
                next_state_node_info ['count'] += 1

            # returns list of edges
            edge_key = 'e_' + str(ps) + '_' + str(a) + '_' + str(ns)
            egde_info = None
            edge = None
            if (edge_key in self.edges_s_to_s):
                edge_info = self.edges_s_to_s [edge_key]
                edge = edge_info ['edge']
                edge_info ['count'] += 1
            else:
                edge = pydot.Edge(prev_state_node, next_state_node, label=str(a))
                edge_info = {
                    'id' : edge_key,
                    'action' : str(a),
                    'prev_state' : prev_state_node_info,
                    'next_state' : next_state_node_info,
                    'edge' : edge,
                    'count' : 1,
                    'added' : False
                }
                prev_state_node_info ['edges'][edge_key] = edge_info
                self.edges_s_to_s [edge_key] = edge_info


            prev_state_action_count = float(prev_state_node_info['action_counts'][action_key])
            for affected_edge_key, affected_edge_info in prev_state_node_info ['edges'].items():

                if (affected_edge_info ['action'] == action_key):

                    affected_edge = affected_edge_info ['edge']
                    affected_ps_info = affected_edge_info ['prev_state']
                    affected_ps = affected_ps_info ['node']
                    affected_ns_info = affected_edge_info ['next_state']
                    affected_ns = affected_ns_info ['node']

                    edge_probability = float(affected_edge_info ['count']) / prev_state_action_count
                    edge_exists = edge_probability > 0.1
                    # emerged new edge
                    if (
                        affected_edge_info ['count'] > 5 and
                        edge_exists and
                        not affected_edge_info ['added']
                    ):
                        print ('adding edge: ' + affected_edge_key)
                        if (not affected_ps_info['added']):
                            self.graph.add_node(affected_ps)
                            affected_ps_info ['added'] = True
                        if (not affected_ns_info['added']):
                            self.graph.add_node(affected_ns)
                            affected_ns_info ['added'] = True

                        affected_ps_info ['edges_count'] += 1
                        affected_ns_info ['edges_count'] += 1
                        affected_edge_info ['added'] = True
                        self.graph.add_edge(affected_edge)

                    # edge disappeared
                    elif (
                        edge_exists and
                        affected_edge_info ['added']
                    ):
                        print ('removing edge: ' + affected_edge_key)
                        self.graph.del_edge ((affected_ps, affected_ns))
                        affected_edge_info ['added'] = False
                        affected_ps_info ['edges_count'] -= 1
                        affected_ns_info ['edges_count'] -= 1

                        if (affected_ps_info ['edges_count'] == 0):
                            print ('removing node: ' + affected_ps_info ['id'])
                            self.graph.del_node (affected_ps)
                            affected_ps_info ['added'] = False
                        if (affected_ns_info ['edges_count'] == 0):
                            print ('removing node: ' + affected_ns_info ['id'])
                            self.graph.del_node (affected_ns)
                            affected_ns_info ['added'] = False

            # if (edge_info ['added']):
            #     print (
            #         'edge: ' + edge_key + ' p: ' + str(edge_probability) + ' c: ' + str(edge_info ['count']) +
            #         ' pn: ' + str(ps) + ' c: ' + str(prev_state_node_info ['count']) +
            #         ' ns: ' + str(ns) + ' c: ' + str(next_state_node_info ['count'])
            #     )

            if (prev_state_node_info ['edges_count'] < 0):
                print ('warning: ps edges_count: ' + str(prev_state_node_info ['edges_count']))

            if (next_state_node_info ['edges_count'] < 0):
                print ('warning: ns edges_count: ' + str(next_state_node_info ['edges_count']))

        if (self.graph_show_index == self.graph_show_every):
            self.show_graph ()
            self.graph_show_index = 0
        else:
            self.graph_show_index += 1

    def show_graph(self):
        png = self.graph.create (format='png')
        file_bytes = np.asarray(bytearray(png), dtype=np.uint8)
        img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
        cv2.imshow ('Associations', img)
        cv2.waitKey (1)
