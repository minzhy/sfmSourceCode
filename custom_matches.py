import numpy as np
import  sqlite3
import cv2
def import_features(images, paths):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    connection.commit()

    # Import the features.
    print('Importing features...')

    for image_name, image_id in tqdm(images.items(), total=len(images.items())):
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name))
        '''
        # if you use opencv orb,you can write this 
        keypoints, des = orb.detectAndCompute(paths.image_path, None)
        kps=[]
        for i in range(len(keypoints)):
            kps.append([keypoints[i].pt[0],keypoints[i].pt[1],keypoints[i].size,keypoints[i].angle]) # opencv orb not Scale invariance       
        '''
       # if you already use other feature and save as .npz,you can load it
        keypoints = np.load(features_path)['keypoints']
        n_keypoints = keypoints.shape[0]

        # Keep only x, y coordinates.
        keypoints = keypoints[:, : 2]
        # x,y,scale,oritation
        keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(
            np.float32)

        keypoints_str = keypoints.tostring()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))
    connection.commit()
    # Close the connection to the database.
    cursor.close()
    connection.close()

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2

# you can match feature use opencv flann
def match(desc1, desc2):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()
    # Match the features and insert the matches in the database.
    print('Matching...')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pair = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 1)
    lt = [(i[0].distance, i[0].queryIdx, i[0].trainIdx) for i in pair]
    matches=np.array(sorted(lt))[:,1:].astype(np.int16)

    matches_str = matches.tostring()
    cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                   (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
# Close the connection to the database.
    connection.commit()
    cursor.close()
    connection.close()
def add_two_view_geometry(self, image_id1, image_id2, matches,
                          F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
    '''
       :param config: configuration of two-view geometry [1-DEGENERATE, 2-CALIBRATED, 3-UNCALIBRATED...] - int
    '''

    assert(len(matches.shape) == 2)
    assert(matches.shape[1] == 2)

    if image_id1 > image_id2:
        matches = matches[:,::-1]

    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    F = np.asarray(F, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    self.execute(
        "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pair_id,) + matches.shape + (array_to_blob(matches), config,
         array_to_blob(F), array_to_blob(E), array_to_blob(H)))
