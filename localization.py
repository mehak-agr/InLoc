import cv2
import time

from graph import Graph
from elements import ImgObj, DistinctFrames, PossibleEdge
from locations import video_dir
import video_utils, display_frame, matcher, vision_api

FRAC_MATCH_THRESH = 0.2


class Localization:
    def __init__(self, graph_obj: Graph):
        self.confirmed_path = []  # contains identity of first node
        self.probable_path = None  # contains PossibleEdge object of current edge
        self.possible_edges = []  # list of PossibleEdge objects
        # It contains current 'src_dest' edge, edges with source as 'dest' and edges with destination as 'src'
        self.next_possible_edges = []  # possible_edges for the upcoming (next) query frame

        self.graph_obj = graph_obj
        self.query_objects = DistinctFrames()  # list - but can be changed later to just store the latest frame
        self.last_5_matches = []  # last 5 matches as (edge_index_matched, edge_name)

        '''
        self.max_confidence_edges: Number of edges with max confidence i.e. those which are to be checked first
        Usually it is equal to 1 and corresponds to the current edge (self.probable_path)
        But can also be equal to 2 when the next edge is almost straight (<20 deg) from current edge and end of current edge is near
        Also self.possible_edges is arranged in such a way that the edges corresponding to max confidence are appended first in the list
        '''
        self.max_confidence_edges = 0
        self.current_location_str = ''
        
    def get_query_params(self, frame_index):
        # Returns params of particular imgObj of query DistinctFrames object for SURF matching
        return self.query_objects.get_object(frame_index).get_elements()
    
    def match_edges(self, query_index):
        # Finds matches of query frame with frames in possible edges and updates last 5 matches
        # Assume all possible edge objects are there in possible_edges
        progress = False
        # match : edge_index (int), maxedge: edge_name(str), maxmatch: fraction_matched(float)
        match, maxmatch, maxedge = None, 0, None  # correspond to best match for given query_index frame

        for i, possible_edge in enumerate(self.possible_edges):
            for j in range(possible_edge.to_match_params[0], possible_edge.to_match_params[1]):
                fraction_matched, features_matched = matcher.SURF_returns(possible_edge.get_frame_params(j),
                                                                          self.get_query_params(query_index))
                if fraction_matched > FRAC_MATCH_THRESH or features_matched > 200:
                    progress = True
                    if fraction_matched > maxmatch:
                        match, maxmatch, maxedge = j, fraction_matched, possible_edge.name

            # First check best match in the max confidence edges. If yes, then no need to check others
            if i == self.max_confidence_edges - 1 and match is not None:
                print(f'---Max match for {query_index}: ({match}, {maxedge})')
                if match is None:
                    self.current_location_str = f'---Max match for {query_index}: (None, None)'
                else:
                    self.current_location_str = f'---Max match for {query_index}: ({match}, {maxedge})'
                self.graph_obj.display_path(0, self.current_location_str)
                # Update last_5_matches
                self.last_5_matches.append((match, maxedge))
                if len(self.last_5_matches) > 5:
                    self.last_5_matches.remove(self.last_5_matches[0])
                return progress

        print(f'---Max match for {query_index}: ({match}, {maxedge})')
        if match is None:
            self.current_location_str = f'---Max match for {query_index}: (None, None)'
        else:
            self.current_location_str = f'---Max match for {query_index}: ({match}, {maxedge})'
        self.graph_obj.display_path(0, self.current_location_str)
        # Update last_5_matches
        self.last_5_matches.append((match, maxedge))
        if len(self.last_5_matches) > 5:
            self.last_5_matches.remove(self.last_5_matches[0])
        return progress
    
    def handle_edges(self):
        # Updates possible_edges, next_possible_edges and decides most_occuring_edge and cur_edge_index based on last_5_matches

        # If self.confirmed_path is empty then starting point is not defined yet
        if len(self.confirmed_path) == 0:
            # Append all edges in self.possible_edges with the to_match_params being only the first frame of each edge
            for node in self.graph_obj.Nodes[0]:
                for edge in node.links:
                    possible_edge_node = PossibleEdge(edge)
                    possible_edge_node.to_match_params = (0, 1)
                    # Change above to include more frames of each edge in determination of initial node
                    self.possible_edges.append(possible_edge_node)

            # Pick up the last query index
            query_index = self.query_objects.no_of_frames() - 1
            progress = self.match_edges(query_index)

            # We need at least 2 (can be changed) matches to consider first node
            if not progress or len(self.last_5_matches) < 2:
                return

            # To find the most occuring edge in last_5_matches
            last_5_edges_matched = []
            for i in range(len(self.last_5_matches)):
                if self.last_5_matches[i][1] is not None:
                    last_5_edges_matched.append(self.last_5_matches[i][1])

            maxCount, most_occuring_edge, most_occuring_second = 0, None, None
            for edge in last_5_edges_matched:
                coun = last_5_edges_matched.count(edge)
                if coun > maxCount:
                    most_occuring_edge = edge
                    most_occuring_second = None
                elif coun == maxCount and edge != most_occuring_edge:
                    most_occuring_second = edge

            # If most_occuring_second is not None it implies 2 edges have max count
            if most_occuring_edge is None or most_occuring_second is not None: return

            # At this point we have the most occuring edge
            for possible_edge in self.possible_edges:
                if possible_edge.name == most_occuring_edge:
                    self.probable_path = possible_edge
                    self.probable_path.to_match_params = (0, possible_edge.no_of_frames)
                    self.max_confidence_edges = 1
                    src, dest = most_occuring_edge.split('_')
                    self.confirmed_path = [int(src)]

            # Setting self.next_possible_edges in this order: 1. current edge 2. nearby edges
            self.next_possible_edges = [self.probable_path]
            node = self.graph_obj.get_node(self.probable_path.edge.dest)
            for edge in node.links:
                present = False
                for possible_edg in self.next_possible_edges:
                    if possible_edg.name == edge.name:
                        present = True
                        break
                if present: continue
                possibleEdge = PossibleEdge(edge)
                self.next_possible_edges.append(possibleEdge)
            node = self.graph_obj.get_node(self.probable_path.edge.src)
            for edge in node.links:
                if edge.dest == self.probable_path.edge.dest:
                    continue
                possibleEdge = PossibleEdge(edge)
                self.next_possible_edges.append(possibleEdge)

        # If something is already there is self.next_possible_edges, use that
        elif len(self.next_possible_edges) != 0:
            self.possible_edges = self.next_possible_edges

        # Else use the node identity stored in self.confirmed_path
        elif len(self.possible_edges) == 0:
            if type(self.confirmed_path[-1]) == int:
                identity = self.confirmed_path[-1]
                node = self.graph_obj.get_node(identity)
                if node is not None:
                    for edge in node.links:
                        possible_edge = PossibleEdge(edge)
                        self.possible_edges.append(possible_edge)

        query_index = self.query_objects.no_of_frames() - 1
        progress = self.match_edges(query_index)

        if not progress: return

        if len(self.last_5_matches) < 5:
            self.next_possible_edges = self.possible_edges
            return

        # To find the most occuring edge in last_5_matches
        last_5_edges_matched = []
        for i in range(len(self.last_5_matches)):
            if self.last_5_matches[i][1] is not None:
                last_5_edges_matched.append(self.last_5_matches[i][1])
        maxCount, most_occuring_edge, most_occuring_second = 0, None, None
        for edge in last_5_edges_matched:
            coun = last_5_edges_matched.count(edge)
            if coun > maxCount:
                most_occuring_edge = edge
                most_occuring_second = None
                maxCount = coun
            elif coun == maxCount and edge != most_occuring_edge:
                most_occuring_second = edge

        # If most_occuring_second is not None it implies 2 edges are having max count
        if most_occuring_edge is None or most_occuring_second is not None: return

        if (None, None) in self.last_5_matches and maxCount < 3: return

        # At this point we have the most occuring edge
        for possible_edge in self.possible_edges:
            if possible_edge.name == most_occuring_edge:
                self.probable_path = possible_edge
                self.max_confidence_edges = 1

        # Finding the most occuring edge index (in the last 5 matches) on the current edge
        edge_indexes = []
        for matches in self.last_5_matches:
            if matches[1] == most_occuring_edge:
                edge_indexes.append(matches[0])
        cur_edge_index = -1  # should hold most occuring edge index (in the last 5 matches) on the current edge
        maxCount = 0
        for index in edge_indexes:
            coun = edge_indexes.count(index)
            if coun > maxCount or (coun == maxCount and index > cur_edge_index):
                cur_edge_index = index
                maxCount = coun

        # Setting self.next_possible_edges in this order:
        # 1. current edge
        # 2. Edge with src as dest of current edge (angle < 20 deg deviated from current edge) added only if cur_edge_index is the last index of current edge
        # 3. Other nearby edges
        self.next_possible_edges = [self.probable_path]
        node = self.graph_obj.get_node(self.probable_path.edge.dest)
        if cur_edge_index > self.probable_path.no_of_frames - 2:
            count_of_straight_edges, straightPossibleEdge = 0, None
            for tup in self.probable_path.edge.angles:
                if abs(tup[1]) < 20:
                    count_of_straight_edges += 1
                    src, dest = tup[0].split('_')
                    edg = self.graph_obj.get_edge(int(src), int(dest))
                    possible_edge = PossibleEdge(edg)
                    straightPossibleEdge = possible_edge
                    self.next_possible_edges.append(possible_edge)
                    self.max_confidence_edges += 1
            if count_of_straight_edges == 1:  # Setting next_pos
                # If cur_edge_index is last index of current edge, and
                # If only one edge is straight ahead (angle < 20 deg) and its first frame matches
                # then the next edge is set as self.probable_path (i.e., it is set as the current edge)
                fraction_matched, features_matched = matcher.SURF_returns(straightPossibleEdge.get_frame_params(0),
                                                                          self.get_query_params(query_index))
                if fraction_matched >= 0.1:
                    self.probable_path = straightPossibleEdge
                    cur_edge_index = 0
                    self.next_possible_edges = [self.probable_path]
                    node = self.graph_obj.get_node(self.probable_path.edge.dest)

        for edge in node.links:
            present = False
            for possible_edg in self.next_possible_edges:
                if possible_edg.name == edge.name:
                    present = True
                    break
            if present: continue
            possibleEdge = PossibleEdge(edge)
            self.next_possible_edges.append(possibleEdge)
        node = self.graph_obj.get_node(self.probable_path.edge.src)

        for edge in node.links:
            if edge.dest == self.probable_path.edge.dest: continue
            possibleEdge = PossibleEdge(edge)
            self.next_possible_edges.append(possibleEdge)

        # Displaying current location on graph
        edgeObj, allow = None, True
        for node in self.graph_obj.Nodes[0]:
            if not allow: break
            for edge in node.links:
                if edge.name == most_occuring_edge:
                    edgeObj = edge
                    allow = False
                    break

        last_jth_matched_img_obj = edgeObj.distinct_frames.get_object(cur_edge_index)
        time_stamp = last_jth_matched_img_obj.get_time()
        total_time = edgeObj.distinct_frames.get_time()
        fraction = time_stamp / total_time if total_time != 0 else 0
        self.graph_obj.on_edge(edgeObj.src, edgeObj.dest, fraction)
        self.graph_obj.display_path(0, self.current_location_str)
        return

    def perform_query(self, video, frames_skipped = 0, livestream = False, write_to_disk = False, folder = None):
        # Receives and reads query video, generates non-blurry gray image frames, creates ImgObj and updates query_objects

        if write_to_disk:
            if folder.exists():
                print(f'---INPUT REQD----{folder} alongwith its contents will be deleted. Continue? (y/n)')
                if input() == 'y':
                    shutil.rmtree(folder)
            (folder / 'jpg').mkdir(exist_ok=True)

        if not livestream:
            video_path = video_dir / 'query_video' / video
        else:
            video_path = video  # should be a URL here
        cap = cv2.VideoCapture(str(video_path))

        frames_skipped += 1
        hessian_threshold = 2500
        detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

        i = 0
        start = time.time()
        while True:
            if livestream:
                cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if i % frames_skipped != 0:
                i += 1
                continue
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if video_utils.is_blurry_grayscale(gray): continue
            break_video = display_frame.run_query_frame(gray)

            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if len(keypoints) < 50:
                print(f'Frame skipped as keypoints {len(keypoints)} less than 50.')
                i += 1
                continue

            text, objects = vision_api.mark_items(gray)
            # text, objects = '', []
            a = (len(keypoints), descriptors, video_utils.serialize_keypoints(keypoints), gray.shape, text, objects)
            img_obj = ImgObj(a[0], a[1], i, a[2], a[3], a[4], a[5])
            self.query_objects.add_img_obj(img_obj)

            if write_to_disk:
                with open(folder / f'image{i}.pkl', 'wb') as output_wb:
                    pickle.dump(img_obj, output_wb, pickle.HIGHEST_PROTOCOL)
                cv2.imwrite(folder / 'jpg' / f'image{i}.jpg', gray)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or break_video:
                break
            self.handle_edges()
            i += 1
            print(f'Time Taken: {(time.time() - start) / i} secs', flush=True)

        cap.release()
        cv2.destroyAllWindows()
