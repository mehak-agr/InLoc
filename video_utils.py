import pickle
import cv2

from elements import ImgObj, DistinctFrames
import matcher
import vision_api

MIN_FRAMES = 50
FRAC_MATCH_THRESH = 0.25
RATIO_THRESH = 0.5


def is_blurry_colorful(image):
    im1, _, _ = cv2.split(image)
    return (cv2.Laplacian(im1, cv2.CV_64F).var() < 100)


def is_blurry_grayscale(gray_image):
    lap = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return (lap < 100)


def serialize_keypoints(keypoints):
    index = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)
    return index


def deserialize_keypoints(index):
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


def save_distinct_ImgObj(video_str, folder, frames_skipped: int = 0, check_blurry: bool = True,
                         hessian_threshold: int = 2500, ensure_min = True):
    """Saves non redundant and distinct frames of a video in folder
    Parameters
    ----------
    video_str : is video_str = "webcam" then loads webcam. O.W. loads video at video_str location,
    folder : folder where non redundant images are to be saved,
    frames_skipped: Number of frames to skip and just not consider,
    check_blurry: If True then only considers non blurry frames but is slow
    hessian_threshold
    ensure_min: whether a minimum no of frames (at least one per 50) is to be kept irrespective of
        whether they are distinct or not

    Returns
    -------
    array,
        returns array contaning non redundant frames(mat format)
    """

    if video_str == 'webcam':
        video_str = 0
    else:
        video_str = str(video_str)
    cap = cv2.VideoCapture(video_str)
    print('Total Frames: ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    
    frames_skipped += 1
    i, i_prev, i_of_a, a, b = 0, 0, 0, None, None
    check_next_frame = False

    # Make a
    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    text, objects = vision_api.mark_items(gray)
    a = (len(keypoints), descriptors, serialize_keypoints(keypoints), gray.shape, text, objects)
    img_obj = ImgObj(a[0], a[1], i, a[2], a[3], a[4], a[5])

    # Save a
    with open(folder / f'image{i}.pkl', 'wb') as output_wb:
        pickle.dump(img_obj, output_wb, pickle.HIGHEST_PROTOCOL)
    (folder / 'jpg').mkdir(exist_ok=True)
    cv2.imwrite(str(folder / 'jpg' / f'image{i}.jpg'), gray)

    distinct_frames = DistinctFrames()
    distinct_frames.add_img_obj(img_obj)
    
    while True:
        print(i, i_of_a, i_prev, end = '|')
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i % frames_skipped != 0 and not check_next_frame:
                i += 1
                continue
                
            cv2.imshow('frame', gray)

            if check_blurry:
                if is_blurry_grayscale(gray):
                    check_next_frame = True
                    print(f'frame {i} skipped as blurry')
                    i += 1
                    continue
                check_next_frame = False

            # Make b
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            text, objects = vision_api.mark_items(gray)
            b = (len(keypoints), descriptors, serialize_keypoints(keypoints), gray.shape, text, objects)

            if len(keypoints) < 100:
                print(f'frame {i} skipped as {len(keypoints)} <100')
                i += 1
                continue

            image_fraction_matched, min_good_matches = matcher.SURF_returns(a, b, 2500, RATIO_THRESH, True)
            if image_fraction_matched == -1:
                check_next_frame = True
                i += 1
                continue
            check_next_frame = False

            if 0 < image_fraction_matched < FRAC_MATCH_THRESH or min_good_matches < MIN_FRAMES or (ensure_min and i - i_prev > MIN_FRAMES):
                print(f'{image_fraction_matched} fraction match between {i_of_a} and {i}')

                img_obj2 = ImgObj(b[0], b[1], i, b[2], b[3], b[4], b[5])
                distinct_frames.add_img_obj(img_obj2)
                a, i_of_a, i_prev = b, i, i

                # Save b
                with open(folder / f'image{i}.pkl', 'wb') as output_wb:
                    pickle.dump(img_obj2, output_wb, pickle.HIGHEST_PROTOCOL)
                cv2.imwrite(str(folder / 'jpg' / f'image{i}.jpg'), gray)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            break
    print('Created distinct frames object')
    cap.release()
    cv2.destroyAllWindows()
    distinct_frames.calculate_time()
    return distinct_frames
