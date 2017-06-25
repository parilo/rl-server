import pydot
import cv2
import numpy as np
import math
import random
import copy

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
                    'action_counts' : {},
                    'value' : 0, # value function
                    'avg_reward' : 0
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
                    'action_counts' : {},
                    'value' : 0, # value function
                    'avg_reward' : 0
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
                    'added' : False,
                    'probablity' : 0.0
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
                    affected_edge_info ['probablity'] = edge_probability
                    edge_exists = edge_probability > 0.1
                    # emerged new edge
                    if (
                        affected_edge_info ['count'] > 5 and
                        edge_exists and
                        not affected_edge_info ['added']
                    ):
                        print ('adding edge: ' + affected_edge_key)
                        if (not affected_ps_info['added']):
                            # print ('adding node: ' + affected_ps_info ['id'])
                            self.graph.add_node(affected_ps)
                            affected_ps_info ['added'] = True
                        if (not affected_ns_info['added']):
                            # print ('adding node: ' + affected_ns_info ['id'])
                            self.graph.add_node(affected_ns)
                            affected_ns_info ['added'] = True

                        affected_ps_info ['edges_count'] += 1
                        affected_ns_info ['edges_count'] += 1
                        affected_edge_info ['added'] = True
                        self.graph.add_edge(affected_edge)

                    # edge disappeared
                    elif (
                        not edge_exists and
                        affected_edge_info ['added']
                    ):
                        print ('removing edge: ' + affected_edge_key)
                        edge_list = self.graph.get_edge (affected_edge.get_source (), affected_edge.get_destination ())
                        if (isinstance (edge_list, (pydot.Edge))):
                            self.graph.del_edge (affected_ps, affected_ns)
                        elif (isinstance (edge_list, (list))):
                            for i in range (len(edge_list)):
                                if (
                                    edge_list [i].obj_dict['attributes']['label'] == affected_edge.obj_dict['attributes']['label']
                                ):
                                    self.graph.del_edge (affected_ps, affected_ns, index=i)
                                    break
                        else:
                            print ('--- got: ' + str(edge_list))

                        affected_edge_info ['added'] = False
                        affected_ps_info ['edges_count'] -= 1
                        affected_ns_info ['edges_count'] -= 1

                        if (affected_ps_info ['edges_count'] == 0):
                            # print ('removing node: ' + affected_ps_info ['id'])
                            self.graph.del_node (affected_ps)
                            affected_ps_info ['added'] = False
                        if (affected_ns_info ['edges_count'] == 0):
                            # print ('removing node: ' + affected_ns_info ['id'])
                            self.graph.del_node (affected_ns)
                            affected_ns_info ['added'] = False

                    # elif ()

            # recalculate value function of prev_state and next_state
            self.recalc_value_function (next_state_node_info, r)
            self.recalc_value_function (prev_state_node_info, r)


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
            print ('-----------------------------------------------')
            # for n in self.state_nodes:
            #     if (n is not None):
            #         if (n ['edges_count'] > 0 or n ['added']):
            #             print (n ['id'] + ' v: ' + str(n['value']))
            #             # print (n ['id'] + ' added: ' + str(n['added']) + ' edges_count: ' + str(n['edges_count']))
            # print ('-----------------------')
            # for ek, ei in self.edges_s_to_s.items():
            #     # if (ei['added']):
            #     print (ek + ' c: ' + str(ei['count']) + ' p: ' + str(ei['probablity']))
            print ('--- edges: ' + str(len(self.edges_s_to_s.items())))
            print ('-----------------------------------------------')

            self.sparse_not_added_edges ()
            self.recolor_state_nodes ()
            self.show_graph ()
            self.graph_show_index = 0
        else:
            self.graph_show_index += 1

    def sparse_not_added_edges (self):
        for ek, ei in self.edges_s_to_s.copy().items():
            if (not ei['added']):
                if (ei ['count'] == 1):
                    self.del_edge (ei)
                else:
                    ei ['count'] -= 1

    def del_edge (self, edge_info):
        edge_key = edge_info ['id']
        # print ('deleting edge: ' + edge_key + ' p: ' + str(edge_info ['probablity']) + ' c: ' + str(edge_info ['count']))
        del edge_info ['prev_state']['edges'][edge_key]
        del self.edges_s_to_s [edge_key]

    def recalc_value_function (self, state_info, reward):
        # state_info ['avg_reward'] +=
        value = reward
        # print ('--- calc value')
        for edge_key, edge_info in state_info ['edges'].items():
            # print (' add: ' + str(edge_info ['probablity']) + ' ' + str(edge_info ['next_state']['value']))
            value += edge_info ['probablity'] * edge_info ['next_state']['value']
        c = len (state_info ['edges'])
        # print (' final: ' + str(value) + ' c: ' + str(c))
        # renormalization of edges probablities
        if (c > 0):
            value /= c
        state_info ['value'] = value

    def recolor_state_nodes (self):
        # min_val = 0.0
        max_val = 0.0
        for n in self.state_nodes:
            if (n is not None):
                if (n ['added']):
                    v = math.fabs (n ['value'])
                    if (v > max_val):
                        max_val = v
                    # if (v < min_val):
                    #     min_val = v

        # print ('--- recolor max val: ' + str(max_val))
        for n in self.state_nodes:
            if (n is not None):
                if (n ['added']):
                    v = n ['value']
                    intensity_v = 255 - int (math.floor(255 * (math.fabs(v) / max_val)))
                    intensity = '{:02x}'.format(intensity_v)
                    color = '#ffffff'
                    if (v < 0):
                        color = '#ff' + intensity + intensity
                    else:
                        color = '#' + intensity + 'ff' + intensity
                    n ['node'].obj_dict['attributes']['fillcolor'] = color
                    # print (n ['id'] + ' v: ' + str(v) + ' c: ' + color)


    def show_graph(self):
        png = self.graph.create (format='png')
        file_bytes = np.asarray(bytearray(png), dtype=np.uint8)
        img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
        cv2.imshow ('Associations', img)
        cv2.waitKey (1)

    def control (self, current_states_batch):
        act_centroids = self.som_action.get_centroids ()
        if (act_centroids is None):
            return None

        states_clusters = self.som_state.get_clusters (current_states_batch)
        actions = []
        for s in states_clusters:
            n = self.state_nodes [s]
            found_better_next_state = False
            if (n is not None):
                max_v = n ['value']
                action = None
                for edge_key, edge_info in n ['edges'].copy().items():
                    if (edge_info ['added']):
                        ns = edge_info ['next_state']
                        ns_v = ns ['value']
                        if (ns ['added'] and ns_v > max_v):
                            found_better_next_state = True
                            max_v = ns_v
                            action = edge_info ['action']

            if (found_better_next_state):
                a = act_centroids [int(action)]
                # print ('--- associations control')
                # print (a)
                actions.append (a)
            else:
                a = act_centroids [random.randint (0, self.som_action.get_clusters_count() - 1)]
                # print ('--- associations explore')
                # print (a)
                actions.append (a)

        return np.array(actions).tolist ()



    # def find_path_to_better_state (self, going_paths, ended_paths = [], visited_states = {})
    #     new_going_paths = []
    #     for p in going_paths:
    #         last_state = p ['last_state']
    #         path_is_ended = True
    #         for edge_key, edge_info in last_state ['edges'].items():
    #             if (edge_info ['added']):
    #                 ns = edge_info ['next_state']
    #                 if (ns ['added'] and not ns ['id'] in visited_states):
    #                     path_is_ended = False
    #                     visited_states [ns['id']] = True
    #                     new_path = {
    #                         'last_state' : ns,
    #                         'path_steps' : p ['path_steps'] + [ns]
    #                     }
    #                     new_going_paths.append (new_path)
    #
    #         if (path_is_ended):
    #             ended_paths.append (p)
    #
    #     if (len(new_going_paths) > 0):
    #         self.find_path_to_better_state (self, ended_paths)
