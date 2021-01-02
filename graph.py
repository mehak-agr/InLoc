import copy
import cv2
import pickle
from pathlib import Path

from elements import Node, Edge, FloorMap, DistinctFrames
from video_utils import save_distinct_ImgObj
from locations import pickle_dir, video_dir
import display_frame

class Graph:
    def __init__(self):
        self.new_node_index = 0
        self.Nodes = []  # list of list of Nodes Nodes[0] will be list of all Nodes of floor0
        self.no_of_floors = 0
        self.Floor_map = []
        self.path_traversed = []

    # private functions
    def get_node(self, identity, z = None):
        if z is not None:
            for node in self.Nodes[z]:
                if identity == node.identity:
                    return node
        else:
            for floor_nodes in self.Nodes:
                for node in floor_nodes:
                    if identity == node.identity:
                        return node
        return None
    
    def get_edge(self, src, dest, z_src = None, z_dest = None):
        node_src = self.get_node(src, z_src)
        node_dest = self.get_node(dest, z_dest)
        if node_src is not None and node_dest is not None:
            for edge in node_src.links:
                if edge.dest == dest:
                    return edge
        return None
    
    def get_edges(self, identity: int, z=None):
        node = self.get_node(identity, z)
        if node is not None:
            return node.links
        return None
    
    def _create_node(self, name, x, y, z):
        identity = self.new_node_index
        node = Node(identity, name, x, y, z)
        self._add_node(node)
        
    def _add_node(self, node):
        z = node.coordinates[2]
        if len(self.Nodes) <= z:
            for i in range(0, z + 1 - len(self.Nodes)):
                self.Nodes.append([])
        if isinstance(node, Node):
            if node not in self.Nodes[z]:
                if isinstance(node.links, list):
                    if len(node.links) == 0 or isinstance(node.links[0], Edge):
                        self.Nodes[z].append(node)
                        self.new_node_index = self.new_node_index + 1
                else:
                    raise Exception('node.links is not a list of Edge')
            else:
                raise Exception('node is already present')
        else:
            raise Exception('node format is not of Node')

    def _nearest_node(self, x, y, z):
        def distance(node):
            delx = abs(node.coordinates[0] - x)
            dely = abs(node.coordinates[1] - y)
            return delx ** 2 + dely ** 2

        minimum, nearest_node = -1, None
        for node in self.Nodes[z]:
            if abs(node.coordinates[0] - x) < 50 and abs(node.coordinates[1] - y) < 50:
                if minimum == -1 or distance(node) < minimum:
                    nearest_node = node
                    minimum = distance(node)
        return nearest_node
    
    def _connect(self, node_src, node_dest):
        if isinstance(node_src, Node) and isinstance(node_dest, Node):
            if node_dest.identity < self.new_node_index and node_src.identity < self.new_node_index:
                edge = Edge(True, node_src.identity, node_dest.identity)
                node_src.links.append(edge)
            else:
                raise Exception('Wrong identities of Nodes')
        else:
            raise Exception('<node> format is not of Node, or is already present')

    def _delete_node(self, node: Node):
        z = node.coordinates[2]
        if node in self.Nodes[z]:
            self.Nodes[z].remove(nd)
            for floor_Nodes in self.Nodes:
                for node in floor_Nodes:
                    for edge in node_dest.links:
                        if node.identity == edge.dest:
                            node.links.remove(edge)
        else:
            raise Exception('<node> does not exists in Nodes')
            
    def _get_edge_slope(self, edge: Edge, floor: int = 0):
        src = edge.src
        dest = edge.dest
        src_node = None
        dest_node = None
        for node in self.Nodes[floor]:
            if node.identity == src:
                src_node = node
            if node.identity == dest:
                dest_node = node
        src1 = src_node.coordinates[0]
        src2 = src_node.coordinates[1]
        dest1 = dest_node.coordinates[0]
        dest2 = dest_node.coordinates[1]
        slope_in_degree = None

        if (dest1 - src1) == 0:
            slope_in_degree = 90
        elif (-1) * (dest2 - src2) > 0 and (dest1 - src1) > 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
        elif (-1) * (dest2 - src2) > 0 and (dest1 - src1) < 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 180 + slope_in_degree
        elif (-1) * (dest2 - src2) < 0 and (dest1 - src1) < 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 180 + slope_in_degree - 360
        elif (-1) * (dest2 - src2) < 0 and (dest1 - src1) > 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 360 + slope_in_degree - 360
        else:
            print('In: Graph._get_edge_slope')
            print('no such case exists')

        return slope_in_degree
    
    def _get_angle_between_two_edges(self, edge1: Edge, edge2: Edge, floor: int = 0):
        print('In: Graph._get_angle_between_two_edges')
        
        slope1 = self._get_edge_slope(edge1, floor)
        print(edge1.name + str(slope1))
        slope2 = self._get_edge_slope(edge2, floor)
        print(edge2.name + str(slope2))

        slope_diff = slope2 - slope1
        if slope_diff > 180:
            slope_diff = slope_diff - 360
        if slope_diff < (-180):
            slope_diff = slope_diff + 360

        return slope_diff
    
    def _set_specific_edge_angles(self, cur_edge: Edge):
        print('In: Graph._set_specific_edge_angles')
        
        cur_edge.angles = []
        node = self.get_node(cur_edge.dest)
        for next_edge in node.links:
            if next_edge.dest == cur_edge.src:
                ang = 180
            else:
                ang = self._get_angle_between_two_edges(cur_edge, next_edge)
            cur_edge.angles.append((next_edge.name, ang))
        print(cur_edge.name)
        print(cur_edge.angles)
        
    def _set_all_angles(self, floor_num = 0):
        for node in self.Nodes[floor_num]:
            for edge in node.links:
                self._set_specific_edge_angles(edge)
                
    def _add_edge_images(self, id1: int, id2: int, distinct_frames: DistinctFrames, z1 = None, z2 = None):
        if id1 > self.new_node_index or id2 > self.new_node_index:
            raise Exception('Wrong id passed')
        if not isinstance(distinct_frames, DistinctFrames):
            raise Exception('Invalid parameter for distinct_frames')
        edge = self.get_edge(id1, id2, z1, z2)
        if edge is not None:
            edge.distinct_frames = distinct_frames
            edge.video_length = distinct_frames.get_time()
            return
        raise Exception(f'Edge from {id1} to {id2} not found')
        
    def _add_node_images(self, identity, node_images, z = None):
        if not isinstance(node_images, DistinctFrames):
            raise Exception('node_images is not DistinctFrames object')
        node = self.get_node(identity, z)
        if node is not None:
            node.node_images = node_images
            return
        raise Exception(f'Node {identity} not found!')

    def _add_node_data(self, identity: int, path_of_video: Path, folder_to_save: Path = None, frames_skipped: int = 0,
                       check_blurry: bool = True, hessian_threshold: int = 2500, z_node=None):
        distinct_frames = save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold, ensure_min=True)
        self._add_node_images(identity, distinct_frames, z_node)

    def _add_edge_data(self, id1: int, id2: int, path_of_video: Path, folder_to_save: Path = None, frames_skipped: int = 0,
                       check_blurry: bool = True, hessian_threshold: int = 2500, z1 = None, z2 = None):
        distinct_frames = save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold, ensure_min=True)
        self._add_edge_images(id1, id2, distinct_frames, z1, z2)
        
    def _get_floor_img(self, z, params):
        for floor in self.Floor_map:
            if floor.floor_num== z:
                if params == 'pure':
                    return floor.pure
                elif params == 'impure':
                    return floor.impure
                else:
                    raise Exception('Invalid parameters passed')
        raise Exception('Could not find floor')

    def _set_floor_img(self, z, params, img):
        for floor in self.Floor_map:
            if floor.floor_num== z:
                if params == 'pure':
                    floor.pure = img
                    return
                elif params == 'impure':
                    floor.impure = img
                    return
                else:
                    raise Exception('Invalid parameters')
        raise Exception('Could not find floor')
        
    # public functions
    def print_graph(self, z):
        pure = self._get_floor_img(z, 'pure')
        img = copy.deepcopy(pure)

        for node in self.Nodes[z]:
            img = cv2.circle(
                img, (node.coordinates[0], node.coordinates[1]), 8, (66, 126, 255), -1, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(node.identity), (node.coordinates[0] + 10, node.coordinates[1] + 10), font, 1,
                        (66, 126, 255), 2, cv2.LINE_AA)
            for edge in node.links:
                node_dest = self.get_node(edge.dest, z)
                if node_dest is not None:
                    img = cv2.arrowedLine(img, (node.coordinates[0], node.coordinates[1]),
                                          (node_dest.coordinates[0], node_dest.coordinates[1]), (66, 126, 255), 1,
                                          cv2.LINE_AA)
                else:
                    raise Exception('linkId does not exists')

        cv2.namedWindow(f'Node graph for floor {z}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Node graph for floor {z}', 1600, 1600)
        cv2.imshow(f'Node graph for floor {z}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def print_graph_and_return(self, z):
        pure = self._get_floor_img(z, 'pure')
        img = copy.deepcopy(pure)

        for node in self.Nodes[z]:
            img = cv2.circle(
                img, (node.coordinates[0], node.coordinates[1]), 8, (66, 126, 255), -1, cv2.LINE_AA)
            for edge in node.links:
                node_dest = self.get_node(edge.dest, z)
                if node_dest is not None:
                    img = cv2.line(img, (node.coordinates[0], node.coordinates[1]),
                                   (node_dest.coordinates[0], node_dest.coordinates[1]), (66, 126, 255), 1,
                                   cv2.LINE_AA)
                else:
                    raise Exception('linkId does not exists')
        return img
    
    def mark_nodes(self, z):
        if len(self.Nodes) <= z:
            for i in range(z + 1 - len(self.Nodes)):
                self.Nodes.append([])
        window_text = 'Mark Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                identity = self.new_node_index
                if self._nearest_node(x, y, z) is None:
                    self._create_node('Node-' + str(identity), x, y, z)
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(identity), (x + 10, y + 10), font, 1, (66, 126, 255), 2, cv2.LINE_AA)
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, 'impure')
        img = impure
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def make_connections(self, z):
        node = None
        window_text = f'Make connections for floor {z}'

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global node
                node = self._nearest_node(x, y, z)
            elif event == cv2.EVENT_LBUTTONUP:
                if node is not None:
                    nodecur = self._nearest_node(x, y, z)
                    self._connect(node, nodecur)
                    cv2.arrowedLine(img, (node.coordinates[0], node.coordinates[1]),
                                    (nodecur.coordinates[0], nodecur.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    meanx = (node.coordinates[0] + nodecur.coordinates[0]) // 2
                    meany = (node.coordinates[1] + nodecur.coordinates[1]) // 2
                    cv2.putText(img, f'{node.identity}_{nodecur.identity}', (meanx + 5, meany + 5), font, 0.5,
                                (100, 126, 255), 2, cv2.LINE_AA)
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, 'impure')
        img = impure
        if img is None:
            return
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self._set_all_angles(z)
        
    def delete_nodes(self, z):
        window_text = f'Delete Nodes for floor {z}'

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._nearest_node(x, y, z) is not None:
                    node = self._nearest_node(x, y, z)
                    self._delete_node(node)
                    img = self.print_graph_and_return(z)
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        img = self.print_graph_and_return(z)
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def delete_connections(self, z):
        node = None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global node
                node = self._nearest_node(x, y, z)
            elif event == cv2.EVENT_LBUTTONUP:
                if node is not None:
                    nodecur = self._nearest_node(x, y, z)
                    for edge in node.links:
                        if edge.dest == nodecur.identity:
                            node.links.remove(edge)
                    for edge in nodecur.links:
                        if edge.dest == node.identity:
                            nodecur.links.remove(edge)
                    img = self.print_graph_and_return(z)
                    cv2.namedWindow('Delete connections', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Delete connections', 1600, 1600)
                    cv2.imshow('Delete connections', img)

        img = self.print_graph_and_return(z)
        cv2.namedWindow('Delete connections', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Delete connections', 1600, 1600)
        cv2.imshow('Delete connections', img)
        cv2.setMouseCallback('Delete connections', click_event)
        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()
        
    def read_edges(self, frames_skipped=0, check_blurry=True):
        videos = list((video_dir / 'edge_data').glob('*.webm'))
        videos.sort()
        for video in videos:
            src, dest = video.stem.split('_')
            print('Edge: ', src, dest)
            self._add_edge_data(int(src), int(dest), video, pickle_dir / 'edge_data' / f'edge_{src}_{dest}',
                                frames_skipped, check_blurry)

    def read_nodes(self, frames_skipped=0, check_blurry=True):
        videos = list((video_dir / 'node_data').glob('*.webm'))
        videos.sort()
        for video in videos:
            identity = int(video.stem)
            print('Node: ', identity)
            self._add_node_data(identity, video, pickle_dir / 'node_data' / f'node_{identity}',
                                frames_skipped, check_blurry)
                                    
    def add_floor_map(self, floor_num):
        if floor_num > self.no_of_floors:
            raise Exception(f'Add floor {self.no_of_floors} first!')
        img = cv2.imread(str(video_dir / 'maps' / f'map{floor_num}.jpg'))
        if img is not None:
            floor_map = FloorMap(floor_num, img)
            self.Floor_map.append(floor_map)
            self.no_of_floors = self.no_of_floors + 1
        else:
            raise Exception(f'Cannot read map{floor_num}.jpg image.')

    def on_node(self, identity):
        if len(self.path_traversed) > 0:
            if type(self.path_traversed[-1]) == int:
                prev_node = self.path_traversed[-1]
                edge = self.get_edge(identity, prev_node)
                if edge is not None:
                    self.path_traversed.append((prev_node, identity, 1))
        self.path_traversed.append(identity)
        
    def on_edge(self, src: int, dest: int, fraction_traversed: float = 0.5):
        if len(self.path_traversed) > 0:
            prev = self.path_traversed[-1]
            if type(prev) == tuple:
                if prev[0] == src:
                    if prev[1] == dest and prev[2] > fraction_traversed:
                        return
                    self.path_traversed[-1] = (src, dest, fraction_traversed)
                    return
                else:
                    self.path_traversed[-1] = (prev[0], prev[1], 1)
                    self.path_traversed.append(prev[1])
            if self.path_traversed[-1] != src:
                self.path_traversed.append(src)
        else:
            self.path_traversed.append(src)
        self.path_traversed.append((src, dest, fraction_traversed))
        
    def display_path(self, z, current_location_str = ''):
        img = self.print_graph_and_return(0)
        for item in self.path_traversed:
            if type(item) == int:
                node = self.get_node(item)
                img = cv2.circle(img, (node.coordinates[0], node.coordinates[1]), 10, (150, 0, 0), -1, cv2.LINE_AA)

            if type(item) == tuple:
                src_node = self.get_node(item[0])
                dest_node = self.get_node(item[1])
                start_coordinates = (src_node.coordinates[0], src_node.coordinates[1])
                end_coordinates = (int(src_node.coordinates[0] + item[2] * (dest_node.coordinates[0] - src_node.coordinates[0])),
                                   int(src_node.coordinates[1] + item[2] * (dest_node.coordinates[1] - src_node.coordinates[1])))
                img = cv2.line(img, start_coordinates, end_coordinates, (150, 0, 0), 6, cv2.LINE_AA)

        if len(self.path_traversed) > 0:
            last = self.path_traversed[-1]
            if type(last) == int:
                node = self.get_node(last)
                img = cv2.circle(img, (node.coordinates[0], node.coordinates[1]), 15, (0, 200, 0), -1, cv2.LINE_AA)
            if type(last) == tuple:
                src_node = self.get_node(last[0])
                dest_node = self.get_node(last[1])
                end_coordinates = (int(src_node.coordinates[0] + last[2] * (dest_node.coordinates[0] - src_node.coordinates[0])),
                                   int(src_node.coordinates[1] + last[2] * (dest_node.coordinates[1] - src_node.coordinates[1])))
                img = cv2.circle(img, end_coordinates, 15, (0, 200, 0), -1, cv2.LINE_AA)

        cv2.putText(img, current_location_str, (20, 32), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        display_frame.run_graph_frame(img)

    def save_graph(self, file_name):
        with open(pickle_dir / 'graph_data' / file_name, 'wb') as output_wb:
            pickle.dump(self, output_wb, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_graph(file_name):
        with open(pickle_dir / 'graph_data' / file_name, 'rb') as input_rb:
            graph = pickle.load(input_rb)
        return graph
