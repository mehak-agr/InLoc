import cv2
import numpy as np
import vision_api

graph_frame = None
query_video_frame = None

def run_query_frame(img):
    global query_video_frame
    query_video_frame = img
    return show_frames()

def run_graph_frame(img):
    global graph_frame
    graph_frame = img
    show_frames()

def show_frames():
    global graph_frame
    global query_video_frame

    if graph_frame is not None and query_video_frame is not None:
        w_1, w_2, h = 450 * 1.25, 800 * 1.25, 550 * 1.25
        w_1, w_2, h = int(w_1), int(w_2), int(h)

        # Process graph frame
        graph_frame_resized = cv2.resize(graph_frame, (w_1, h), interpolation = cv2.INTER_AREA)
        # Process query frame
        query_3_channel = cv2.cvtColor(query_video_frame, cv2.COLOR_GRAY2BGR)
        # '''
        query_3_channel, text_desc, object_names = vision_api.mark_items(query_3_channel, return_frame=True)
        text_desc, object_names = vision_api.mark_items(query_3_channel)
        if len(text_desc) != 0:
            print('Text:', text_desc)
        if len(object_names) != 0:
            print('Objects:', object_names)
        # '''
        query_3_channel_resized = cv2.resize(query_3_channel, (w_2, h), interpolation=cv2.INTER_AREA)

        # Put them side by side
        graph_query = np.concatenate((graph_frame_resized, query_3_channel_resized), axis=1)

        # Display
        cv2.imshow('Live Stream and localization', graph_query)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return True
        else:
            return False
