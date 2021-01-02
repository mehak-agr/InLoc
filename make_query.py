from graph import Graph
from localization import Localization

graph_obj = Graph.load_graph('graph_1.pkl')
localization = Localization(graph_obj)
localization.perform_query(video='VID_20190618_202826.webm', frames_skipped = 1)


