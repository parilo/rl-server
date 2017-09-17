import pydot
import cv2
import numpy as np
import math
import random
import copy

# https://github.com/erocarrera/pydot/blob/master/pydot.py

class AssociationsGraph ():

    def __init__ (self, train_loop, cluster_state, cluster_action):

        self.train_loop = train_loop
        self.cluster_state = cluster_state
        self.cluster_action = cluster_action

        slen = self.cluster_state.get_clusters_count ()
        # alen = self.som_action.get_clusters_count ()
        self.state_nodes = [None] * slen
        # self.action_nodes = [None] * alen
        # self.edges_s_to_a = [[None for x in range(alen)] for y in range(slen)]
        # self.edges_a_to_s = [[None for x in range(slen)] for y in range(alen)]
        self.edges_s_to_s = {}

        self.graph = pydot.Dot(graph_type='digraph')
        self.graph_show_index = 0
        self.graph_show_every = 200

        self.act_count = 0
        self.act_control_count = 0

    def process_outputs (self):

        rewards = self.train_loop.store_outputs [1] # from fifo queue
        actions = self.train_loop.store_outputs [2]
        prev_states = self.train_loop.store_outputs [3]
        next_states = self.train_loop.store_outputs [4]

        # print('--- actions')
        # print(actions)
        # print('--- clusters')
        prev_states_clusters = self.cluster_state.get_clusters (prev_states)
        # print(prev_states_clusters)
        next_states_clusters = self.cluster_state.get_clusters (next_states)
        # print(next_states_clusters)
        actions_clusters = self.cluster_action.get_clusters (actions)
        # print(actions_clusters)
        # print('----------')

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
                    'last_rewards' : []
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
                    'value' : 0, # value function,
                    'last_rewards' : []
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
                edge = pydot.Edge(prev_state_node, next_state_node, label=str(a), fillcolor="green")
                edge_info = {
                    'id' : edge_key,
                    'action' : str(a),
                    'prev_state' : prev_state_node_info,
                    'next_state' : next_state_node_info,
                    'edge' : edge,
                    'count' : 1,
                    'added' : False,
                    'probablity' : 0.0,
                    'value' : 0,
                    'last_rewards' : []
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
                    edge_exists = edge_probability > 0.3
                    # emerged new edge
                    if (
                        affected_edge_info ['count'] > 2 and
                        edge_exists and
                        not affected_edge_info ['added']
                    ):
                        # print ('adding edge: ' + affected_edge_key)
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
                        # print ('removing edge: ' + affected_edge_key)
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
            # self.recalc_value_function (next_state_node_info, r)
            # self.recalc_value_function (prev_state_node_info, r)
            self.recalc_value_function_from_frequency (next_state_node_info)
            self.recalc_value_function (edge_info, r)


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
            for n in self.state_nodes:
                if (n is not None):
                    if (n ['added']):
                        print (n ['id'] + ' v: ' + str(n['value']))
                        # print (n ['id'] + ' added: ' + str(n['added']) + ' edges_count: ' + str(n['edges_count']))
            print ('-----------------------')
            for ek, ei in self.edges_s_to_s.items():
                if (ei['added']):
                    print (ek + ' v: ' + str(ei['value']) + ' p: ' + str(ei['probablity']) + ' c: ' + str(ei['count']))
            print ('--- edges: ' + str(len(self.edges_s_to_s.items())))
            print ('-----------------------------------------------')

            self.sparse_not_added_edges ()
            self.recolor_state_nodes ()
            self.recolor_action_nodes ()
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

        state_info ['value'] += 0.05 * reward
        lr = state_info ['last_rewards']
        lr.append (reward)
        if (len(lr) > 20):
            r = lr.pop (0)
            state_info ['value'] -= 0.05 * reward

    def recalc_value_function_from_frequency (self, state_info):
        state_info ['value'] += 0.1

    def recolor_elements (self, elements_list, element_key):
        # min_val = 0.0
        max_val = 0.0
        for n in elements_list:
            if (n is not None):
                if (n ['added']):
                    v = math.fabs (n ['value'])
                    if (v > max_val):
                        max_val = v
                    # if (v < min_val):
                    #     min_val = v

        # print ('--- recolor max val: ' + str(max_val))
        for n in elements_list:
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
                    n [element_key].obj_dict['attributes']['fillcolor'] = color
                    # print (n ['id'] + ' v: ' + str(v) + ' c: ' + color)

    def recolor_state_nodes (self):
        self.recolor_elements (self.state_nodes, 'node')

    def recolor_action_nodes (self):
        self.recolor_elements (self.edges_s_to_s.values (), 'edge')

    # def recolor_state_nodes (self):
    #     # min_val = 0.0
    #     max_val = 0.0
    #     for n in self.state_nodes:
    #         if (n is not None):
    #             if (n ['added']):
    #                 v = math.fabs (n ['value'])
    #                 if (v > max_val):
    #                     max_val = v
    #                 # if (v < min_val):
    #                 #     min_val = v
    #
    #     # print ('--- recolor max val: ' + str(max_val))
    #     for n in self.state_nodes:
    #         if (n is not None):
    #             if (n ['added']):
    #                 v = n ['value']
    #                 intensity_v = 255 - int (math.floor(255 * (math.fabs(v) / max_val)))
    #                 intensity = '{:02x}'.format(intensity_v)
    #                 color = '#ffffff'
    #                 if (v < 0):
    #                     color = '#ff' + intensity + intensity
    #                 else:
    #                     color = '#' + intensity + 'ff' + intensity
    #                 n ['node'].obj_dict['attributes']['fillcolor'] = color
    #                 # print (n ['id'] + ' v: ' + str(v) + ' c: ' + color)


    def show_graph(self):
        return
        png = self.graph.create (format='png')
        file_bytes = np.asarray(bytearray(png), dtype=np.uint8)
        img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
        cv2.imshow ('Associations', img)
        cv2.waitKey (1)

    def has_more_than_one_edge(self, state_cluster_index):
        added_edges = 0
        state = self.state_nodes [state_cluster_index]
        if state is not None:
            for edge_key, edge_info in state ['edges'].copy().items():
                if (edge_info ['added']):
                    added_edges += 1
                    if added_edges > 1:
                        return True
        return False

    def count_edges(self, state_cluster_index):
        added_edges = 0
        state = self.state_nodes [state_cluster_index]
        if state is not None:
            for edge_key, edge_info in state ['edges'].copy().items():
                if (edge_info ['added']):
                    added_edges += 1
        return added_edges

    def control (self, current_states_batch):

        available_actions = self.cluster_action.get_neurons ()
        available_actions_count = self.cluster_action.get_clusters_count()
        batch_size = len(current_states_batch)
        control_actions = np.zeros((batch_size, available_actions.shape[1]))

        states_clusters = self.cluster_state.get_clusters (current_states_batch)
        for i, s in zip(range(batch_size), states_clusters):

            # if not self.has_more_than_one_edge(s):
            #     ai = random.randint(0, available_actions_count-1)
            #     control_actions[i] = available_actions[ai]
            #     continue

            self.act_count += 1

            if self.act_count == 10000:
                print ('--- act stat: {} {} {}'.format(self.act_control_count, self.act_count, float(self.act_control_count)/self.act_count ))
                self.act_count = 0
                self.act_control_count = 0

            edge_count = self.count_edges(s)
            if edge_count == 0:
                ai = random.randint(0, available_actions_count-1)
                control_actions[i] = available_actions[ai]
                continue

            if random.uniform(0, 1) < 0.5 / self.count_edges(s):
                ai = random.randint(0, available_actions_count-1)
                control_actions[i] = available_actions[ai]
                continue

            action_id = self.mc_graph_search(s)
            if action_id is None:
                ai = random.randint(0, available_actions_count-1)
                control_actions[i] = available_actions[ai]
            else:
                # print ('--- control: s: {} a: {}'.format(s, action_id))
                control_actions[i] = available_actions[action_id]
                self.act_control_count += 1

        # print ('--- control actions')
        # print (control_actions)
        return control_actions.tolist ()
        # return np.zeros((current_states_batch.shape[0], 2))

    def mc_graph_search (self, initial_state_cluster):

        search_depth = 10
        num_of_samples = 10
        # print ('--- mcgs from state: {}'.format(initial_state_cluster))
        # need to check every possible actions
        state = self.state_nodes [initial_state_cluster]
        if (state is not None):

            best_action_value = -1000.0
            best_action_id = None

            for edge_key, edge_info in state ['edges'].copy().items():
                if (edge_info ['added']):
                    edge_value = edge_info ['probablity'] * edge_info ['value']
                    next_state = edge_info ['next_state']
                    next_state_id = int(next_state ['id'])
                    action_id = int(edge_info ['action'])
                    value = 0.0
                    depth = 0.0
                    for i in range(num_of_samples):
                        sample_value, sample_depth = self.mc_graph_search_sample (next_state_id, search_depth)
                        value += sample_value / float(num_of_samples)
                        depth += sample_depth / float(num_of_samples)
                    # print ('--- mcgs: a: {} v: {} ev: {} s: {} mcdsv: {} d: {}'.format(action_id, (edge_value + value), edge_value, next_state_id, value, depth))

                    value += edge_value

                    if value > best_action_value:
                        best_action_value = value
                        best_action_id = action_id

            return best_action_id
        else:
            return None

    def mc_graph_search_sample (self, initial_state_cluster, search_depth):

        sample_value = 0
        current_depth = 0
        prev_state = self.state_nodes [initial_state_cluster]
        visited = {prev_state['id'] : True}

        if prev_state is None:
            return sample_value, current_depth

        while True:

            edges_count = len(prev_state ['edges'].values())
            # print ('--- edges count: {}'.format(edges_count))
            if edges_count < 1:
                return sample_value, current_depth

            # try to select next
            edge_value = 0
            edge_info = None
            for i in range(10):
                ei = random.randint(0, edges_count - 1)
                edge_info = list(prev_state ['edges'].values())[ei]
                if (edge_info ['added']):
                    edge_value = edge_info ['probablity'] * edge_info ['value']
                    break
                else:
                    edge_info = None

            if edge_info is None:
                return sample_value, current_depth
            else:
                prev_state = edge_info['next_state']
                if prev_state['id'] in visited:
                    return sample_value, current_depth
                else:
                    sample_value += edge_value
                    current_depth += 1

            if current_depth == search_depth - 1:
                return sample_value, current_depth



    # def control (self, current_states_batch):
    #     act_centroids = self.cluster_action.get_centroids ()
    #     if (act_centroids is None):
    #         return None
    #
    #     states_clusters = self.cluster_state.get_clusters (current_states_batch)
    #     actions = []
    #     for s in states_clusters:
    #         n = self.state_nodes [s]
    #         found_next_state = False
    #         if (n is not None):
    #
    #             found_next_state = False
    #             max_v = -100.0
    #             action = None
    #             for edge_key, edge_info in n ['edges'].copy().items():
    #                 if (edge_info ['added']):
    #                     # ns = edge_info ['next_state']
    #                     # ns_v = ns ['value']
    #                     ns_v = edge_info ['value']
    #                     if (ns_v > max_v):
    #                     # if (ns ['added'] and ns_v > max_v):
    #                         found_next_state = True
    #                         max_v = ns_v
    #                         action = edge_info ['action']
    #
    #         if (found_next_state):
    #             a = act_centroids [int(action)]
    #             # print ('--- associations control')
    #             # print (a)
    #             actions.append (a)
    #         else:
    #             a = act_centroids [random.randint (0, self.cluster_action.get_clusters_count() - 1)]
    #             # print ('--- associations explore')
    #             # print (a)
    #             actions.append (a)
    #
    #     return np.array(actions).tolist ()

    # with exploration
    # def control (self, current_states_batch):
    #     act_centroids = self.som_action.get_centroids ()
    #     if (act_centroids is None):
    #         return None
    #
    #     states_clusters = self.som_state.get_clusters (current_states_batch)
    #     actions = []
    #     for s in states_clusters:
    #         n = self.state_nodes [s]
    #         found_better_next_state = False
    #         if (n is not None):
    #             max_v = n ['value']
    #             action = None
    #             for edge_key, edge_info in n ['edges'].copy().items():
    #                 if (edge_info ['added']):
    #                     ns = edge_info ['next_state']
    #                     ns_v = ns ['value']
    #                     if (ns ['added'] and ns_v > max_v):
    #                         found_better_next_state = True
    #                         max_v = ns_v
    #                         action = edge_info ['action']
    #
    #         if (found_better_next_state):
    #             a = act_centroids [int(action)]
    #             # print ('--- associations control')
    #             # print (a)
    #             actions.append (a)
    #         else:
    #             a = act_centroids [random.randint (0, self.som_action.get_clusters_count() - 1)]
    #             # print ('--- associations explore')
    #             # print (a)
    #             actions.append (a)
    #
    #     return np.array(actions).tolist ()



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
