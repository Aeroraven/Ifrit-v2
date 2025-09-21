# Write a python script that parse and visualize DOT file

PATH = r"D:/Project2/Ifrit-v2/rendergraph.dot"

import pydot
(graph,) = pydot.graph_from_dot_file(PATH)
graph.write_png("rendergraph.png")