#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
pydot example 2
@author: Federico Cáceres
@url: http://pythonhaven.wordpress.com/2009/12/09/generating_graphs_with_pydot
"""
import pydot
import cv2
import numpy as np

# this time, in graph_type we specify we want a DIrected GRAPH
graph = pydot.Dot(graph_type='digraph')

# in the last example, we did no explicitly create nodes, we just created the edges and
# they automatically placed nodes on the graph. Unfortunately, this way we cannot specify
# custom styles for the nodes (although you CAN set a default style for all objects on
# the graph...), so let's create the nodes manually.

# creating nodes is as simple as creating edges!
node_a = pydot.Node("Node A", style="filled", fillcolor="red")
# but... what are all those extra stuff after "Node A"?
# well, these arguments define how the node is going to look on the graph,
# you can find a full reference here:
# http://www.graphviz.org/doc/info/attrs.html
# which in turn is part of the full docs in
# http://www.graphviz.org/Documentation.php

# neat, huh? Let us create the rest of the nodes!
node_b = pydot.Node("Node B", style="filled", fillcolor="green")
node_c = pydot.Node("Node C", style="filled", fillcolor="#0000ff")
node_d = pydot.Node("Node D", style="filled", fillcolor="#976856")

#ok, now we add the nodes to the graph
graph.add_node(node_a)
graph.add_node(node_b)
graph.add_node(node_c)
graph.add_node(node_d)

# and finally we create the edges
# to keep it short, I'll be adding the edge automatically to the graph instead
# of keeping a reference to it in a variable
graph.add_edge(pydot.Edge(node_a, node_b))
graph.add_edge(pydot.Edge(node_b, node_c))
graph.add_edge(pydot.Edge(node_c, node_d))
# but, let's make this last edge special, yes?
graph.add_edge(pydot.Edge(node_d, node_a, label="and back we go again", labelfontcolor="#009933", fontsize="10.0", color="blue"))

# and we are done
# graph.write_png('example2_graph.png')
png = graph.create (format='png')
print (len(png))

file_bytes = np.asarray(bytearray(png), dtype=np.uint8)
img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
print (img.shape)
cv2.imshow ('graph', img)
cv2.waitKey (0)




















# https://graph-tool.skewed.de/static/doc/quickstart.html

# import matplotlib
# # matplotlib.use('gtk3agg')
# matplotlib.use('cairo')
# from matplotlib import pyplot
# # import graph_tool.all
# from graph_tool.all import Graph
# from graph_tool.all import graph_draw
# # pyplot.figure()
# # pyplot.show(block=True) # wait before exit
#
# # import io
# # # from graph_tool.all import Graph
# # import matplotlib
# # matplotlib.use('gtk3agg')
# # import graph_tool.all
# # # from graph_tool.all import graphviz_draw
#
# import cv2
# import numpy as np
# import tempfile
#
#
# g = Graph()
#
# v1 = g.add_vertex()
# v2 = g.add_vertex()
# v3 = g.add_vertex()
#
# e = g.add_edge(v1, v2)
#
# # eprop_dict = g.new_edge_property("object")                # Arbitrary python object.
# # eprop_dict[g.edges().next()] = {"foo": "bar", "gnu": 42}  # In this case, a dict.
#
# # ff = open('asd.png', 'wb')
# # f = io.BytesIO()
# # w = io.BufferedWriter (f)
# # r = io.BufferedReader (f)
# ff = tempfile.SpooledTemporaryFile(max_size=10000000, mode='wb')
#
# graph_draw(
#     g,
#     vertex_text=g.vertex_index,
#     vertex_font_size=18,
#     output_size=(200, 200),
#     output=ff,
#     fmt='png'
#     # output="two-nodes.png"
# )
#
# # pos, gv = graphviz_draw(
# #     g,
# #     # vertex_text=g.vertex_index,
# #     # vertex_font_size=18,
# #     # output_size=(200, 200),
# #     # output=ff,
# #     # fmt='png',
# #     # output="two-nodes.png"
# #
# #     size=(200, 200),
# #     # output=ff,
# #     output_format='png',
# #     return_string=True
# # )
#
# # graph_tool.draw.graphviz_draw(
# # g,
# # pos=None,
# # size=(15, 15),
# # pin=False,
# # layout=None,
# # maxiter=None,
# # ratio='fill',
# # overlap=True,
# # sep=None,
# # splines=False,
# # vsize=0.105,
# # penwidth=1.0,
# # elen=None,
# # gprops={},
# # vprops={},
# # eprops={},
# # vcolor='#a40000',
# # ecolor='#2e3436',
# # vcmap=None,
# # vnorm=True,
# # ecmap=None,
# # enorm=True,
# # vorder=None,
# # eorder=None,
# # output='',
# # output_format='auto',
# # fork=False,
# # return_string=False
# # )
#
# # w.flush()
# # a = r.read()
# a = ff._file.getvalue()
# # a = gv
# print (len(a))
# file_bytes = np.asarray(bytearray(a), dtype=np.uint8)
# img = cv2.imdecode (file_bytes, cv2.IMREAD_COLOR)
# # img = np.zeros ((100,100,3))
# print (img.shape)
# cv2.imshow ('graph', img)
# cv2.waitKey (0)
