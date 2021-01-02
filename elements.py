import copy


class Node:
    def __init__(self, identity: int, name: str, x: int, y: int, z: int):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = []
        self.node_images = None

    def __str__(self):
        return str(self.identity)


class Edge:
    def __init__(self, is_connected: bool, src: int, dest: int, distinct_frames = None,
                 video_length: int = None, angles = None):
        self.src = src
        self.dest = dest
        self.distinct_frames = distinct_frames
        self.video_length = video_length
        self.name = f'{src}_{dest}'
        self.angles = angles   # list of form (edge_name, angle)

    def __str__(self):
        return self.name


class FloorMap:
    def __init__(self, floor_num: int = None, img = None):
        self.floor_num = floor_num
        self.pure = img
        self.impure = copy.deepcopy(img)


class ImgObj:
    def __init__(self, no_of_keypoints, descriptors, time_stamp, serialized_keypoints, shape):
        self.no_of_keypoints = no_of_keypoints
        self.descriptors = descriptors
        self.time_stamp = time_stamp
        self.serialized_keypoints = serialized_keypoints
        self.shape = shape

    def get_elements(self):
        return self.no_of_keypoints, self.descriptors, self.serialized_keypoints, self.shape

    def get_time(self):
        return self.time_stamp


class DistinctFrames:
    def __init__(self):
        self.img_objects = []
        self.time_of_path = None

    def add_img_obj(self, img_obj):
        if not isinstance(img_obj, ImgObj):
            raise Exception('Parameter is not an img object')
        self.img_objects.append(img_obj)

    def add_all(self, list_of_img_objects):
        if isinstance(list_of_img_objects, list):
            if (len(list_of_img_objects) != 0):
                if isinstance(list_of_img_objects[0], ImgObj):
                    self.img_objects = list_of_img_objects
                    return
            else:
                self.img_objects = list_of_img_objects
                return
        raise Exception('Parameter is not a list of img objects')

    def calculate_time(self):
        if len(self.img_objects) != 0:
            start_time = self.img_objects[0].time_stamp
            end_time = self.img_objects[-1].time_stamp
            if isinstance(start_time, int) and isinstance(end_time, int):
                self.time_of_path = end_time - start_time
                return
        raise Exception('Error in calculating time of path')

    def get_time(self):
        if self.time_of_path is None:
            self.calculate_time()
        return self.time_of_path

    def no_of_frames(self):
        return len(self.img_objects)

    def get_objects(self, start_index = 0, end_index = -1):
        if start_index == 0 and end_index == -1:
            return self.img_objects[start_index:end_index]
        if (start_index not in range(0, self.no_of_frames())) or (end_index not in range(0, self.no_of_frames())):
            raise Exception('Invalid start / end indexes')
        if start_index > end_index:
            raise Exception('Start index should be less than or equal to end index')
        return self.img_objects[start_index:end_index]

    def get_object(self, index):
        if index not in range(0, self.no_of_frames()):
            raise Exception('Invalid index')
        return self.img_objects[index]


class PossibleEdge:
    def __init__(self, edge: Edge):
        self.name = edge.name
        self.edge = edge
        self.no_of_frames = edge.distinct_frames.no_of_frames()
        self.to_match_params = (0, self.no_of_frames) # Indexes to be queried in the edge

    def __str__(self):
        return self.name

    def get_frame_params(self, frame_index):
        return self.edge.distinct_frames.get_object(frame_index).get_elements()
