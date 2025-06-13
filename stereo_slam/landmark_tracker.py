import numpy as np

class LandmarkTracker:
    def __init__(self):
        self.landmark_ids = {}
        self.landmark_descriptors = []
        self.landmark_timestamps = []

        self.feature_buffer = {}

        self.feature_to_landmark_id = {}
        self.landmark_id_to_feature_index = {}
        self.landmark_id_to_position = {}
        self.global_embeddings = {}

    def add_features(self, timestamp: int, feature:dict) -> bool:
        if timestamp in self.feature_buffer:
            return False

        self.feature_buffer[timestamp] = feature
        # print(f"feature_buffer: {self.feature_buffer}")
        return True

    def get_features(self, timestamp: int):
        return self.feature_buffer[timestamp]

    def add_global_embeddings(self, timestamp: int, embeddings: np.ndarray):
        # normalize the embeddings
        # shape: (1, 768)
        embeddings = embeddings / np.linalg.norm(embeddings)
        self.global_embeddings[timestamp] = embeddings

    def find_top_similar_embeddings(self, query_timestamp: int, embeddings: np.ndarray) -> int:
        # normalize the embeddings
        embeddings = embeddings / np.linalg.norm(embeddings)
        index_to_timestamp = {index: timestamp for index, timestamp in enumerate(self.global_embeddings.keys()) if timestamp != query_timestamp}
        # both embeddings are normalized, so the dot product is the cosine similarity

        # shape: (N, 768)
        global_embeddings = np.array([value for key, value in self.global_embeddings.items() if key != query_timestamp])
        if global_embeddings.shape[0] == 0:
            return -1
        similarities = global_embeddings @ embeddings.T
        max_index = np.argmax(similarities)
        return index_to_timestamp[max_index]

    def get_keypoint_2d(self, timestamp: int, feature_index: int):
        return self.feature_buffer[timestamp]['keypoints'][feature_index]

    def get_feature(self, timestamp: int):
        return self.feature_buffer[timestamp]

    def add_matches(self, timestamp0: int, timestamp1: int, matches: np.ndarray) -> bool:
        match_size = matches.shape[0]
        for i in range(match_size):
            feature_index0 = matches[i, 0]
            feature_index1 = matches[i, 1]
            
            landmark_id0 = self.try_get_landmark_id(timestamp0, feature_index0)
            landmark_id1 = self.try_get_landmark_id(timestamp1, feature_index1)

            self.merge_landmarks_if_not_overlap(timestamp0, feature_index0, landmark_id0, timestamp1, feature_index1, landmark_id1)


    # get landmark id from feature index and timestamp, create new landmark id if not exists
    def try_get_landmark_id(self, timestamp: int, feature_index: int):
        if timestamp in self.feature_to_landmark_id and feature_index in self.feature_to_landmark_id[timestamp]:
            return self.feature_to_landmark_id[timestamp][feature_index]

        landmark_id = len(self.landmark_ids)

        if timestamp not in self.feature_to_landmark_id:
            self.feature_to_landmark_id[timestamp] = {}
        if feature_index not in self.feature_to_landmark_id[timestamp]:
            self.feature_to_landmark_id[timestamp][feature_index] = landmark_id

        if landmark_id not in self.landmark_id_to_feature_index:
            self.landmark_id_to_feature_index[landmark_id] = {}

        if timestamp not in self.landmark_id_to_feature_index[landmark_id]:
            self.landmark_id_to_feature_index[landmark_id][timestamp] = feature_index
        self.landmark_ids[landmark_id] = True
        return landmark_id
    
    def merge_landmarks_if_not_overlap(self, timestamp0: int, feature_index0: int, landmark_id0: int, timestamp1: int, feature_index1: int, landmark_id1: int):
        if landmark_id0 == landmark_id1:
            return
        
        if landmark_id0 is None or landmark_id1 is None:
            return
        
        landmark_id0_timestamps = self.landmark_id_to_feature_index[landmark_id0]
        landmark_id1_timestamps = self.landmark_id_to_feature_index[landmark_id1]
        # overlap we don't merge
        if landmark_id0_timestamps.keys() & landmark_id1_timestamps.keys():
            return

        if landmark_id0 > landmark_id1:
            for timestamp, feature_index in landmark_id1_timestamps.items():
                self.feature_to_landmark_id[timestamp][feature_index] = landmark_id0
                self.landmark_id_to_feature_index[landmark_id0][timestamp] = feature_index
                self.landmark_ids[landmark_id1] = False
                if landmark_id1 in self.landmark_id_to_position:
                    if landmark_id0 not in self.landmark_id_to_position:
                        self.landmark_id_to_position[landmark_id0] = self.landmark_id_to_position[landmark_id1]
                    del self.landmark_id_to_position[landmark_id1]
            del self.landmark_id_to_feature_index[landmark_id1]
        else:
            for timestamp, feature_index in landmark_id0_timestamps.items():
                self.feature_to_landmark_id[timestamp][feature_index] = landmark_id1
                self.landmark_id_to_feature_index[landmark_id1][timestamp] = feature_index
                self.landmark_ids[landmark_id0] = False
                if landmark_id0 in self.landmark_id_to_position:
                    if landmark_id1 not in self.landmark_id_to_position:
                        self.landmark_id_to_position[landmark_id1] = self.landmark_id_to_position[landmark_id0]
                    del self.landmark_id_to_position[landmark_id0]
            del self.landmark_id_to_feature_index[landmark_id0]

    def get_valid_landmark_ids(self) -> list:
        return [landmark_id for landmark_id, is_valid in self.landmark_ids.items() if is_valid]

    def assigned_points_3d_if_not_values(self, timestamp: int, feature_index: int, point_3d: np.ndarray) -> bool:
        if timestamp not in self.feature_to_landmark_id:
            print(f"timestamp {timestamp} not in feature_to_landmark_id")
            return False
        if feature_index not in self.feature_to_landmark_id[timestamp]:
            print(f"feature_index {feature_index} not in feature_to_landmark_id[{timestamp}]")
            #print(f"feature_to_landmark_id[{timestamp}]: {self.feature_to_landmark_id[timestamp]}")
            return False
        landmark_id = self.feature_to_landmark_id[timestamp][feature_index]
        if landmark_id not in self.landmark_id_to_position:
            self.landmark_id_to_position[landmark_id] = point_3d
            return True
        return True

    def get_landmark_positions(self) -> np.ndarray:
        return np.array([self.landmark_id_to_position[landmark_id] for landmark_id in self.landmark_id_to_position])
    
    def get_projection_relations_and_landmark_position(self, timestamp_to_camera_index: dict, track_min_length: int = 3) -> list:
        landmark_id_to_index = {}

        for index, landmark_id in enumerate(self.landmark_id_to_position.keys()):
            landmark_id_to_index[landmark_id] = index

        landmark_positions = np.zeros((len(landmark_id_to_index), 3))
        projection_relations = []
        for landmark_id in landmark_id_to_index.keys():
            position_index = landmark_id_to_index[landmark_id]
            landmark_positions[position_index, :] = self.landmark_id_to_position[landmark_id]

            landmark_index_to_timestamp = self.landmark_id_to_feature_index[landmark_id]
            track_length = len(landmark_index_to_timestamp)
            if track_length < track_min_length:
                continue

            for timestamp, feature_index in landmark_index_to_timestamp.items():
                if timestamp not in timestamp_to_camera_index:
                    continue
                camera_index = timestamp_to_camera_index[timestamp]
                #print(f"timestamp: {timestamp}, camera_index: {camera_index}, feature_index: {feature_index}")
                #print(f"feature_buffer[{timestamp}]: {self.feature_buffer[timestamp]['keypoints'].shape}")
                projection_relations.append((camera_index, landmark_id_to_index[landmark_id], self.feature_buffer[timestamp]['keypoints'][0,feature_index, :].cpu().numpy()))

        return projection_relations, landmark_positions, landmark_id_to_index

    def update_landmark_positions(self, landmark_positions: np.ndarray, landmark_id_to_index: dict):
        for landmark_id, index in landmark_id_to_index.items():
            self.landmark_id_to_position[landmark_id] = landmark_positions[index]

    def get_statistics(self) -> list:
        tracker_statistics = {}

        for landmark_id in self.landmark_id_to_position.keys():
            if landmark_id not in self.landmark_id_to_feature_index:
                print(f"landmark_id {landmark_id} not in landmark_id_to_feature_index")
                continue

            feature_index = self.landmark_id_to_feature_index[landmark_id]

            tracker_length = len(feature_index)

            if tracker_length not in tracker_statistics:
                tracker_statistics[tracker_length] = 0
            tracker_statistics[tracker_length] += 1
        return tracker_statistics


import unittest 
class TestMathFunctions(unittest.TestCase):

    def setUp(self) -> None:
        self.landmark_tracker = LandmarkTracker()

    def test_add_matches(self) -> None:

        self.landmark_tracker.add_matches(0, 1, np.array([[0, 0]]))
        self.landmark_tracker.assigned_points_3d_if_not_values(0, 0, np.array([1, 2, 3]))

        self.assertEqual(self.landmark_tracker.landmark_ids[0], False)
        self.assertEqual(self.landmark_tracker.landmark_ids[1], True)
        self.assertEqual(len(self.landmark_tracker.landmark_id_to_feature_index[1]), 2)
        self.landmark_tracker.add_matches(0, 2, np.array([[0, 1]]))
        self.assertEqual(self.landmark_tracker.landmark_ids[1], False)
        self.assertEqual(self.landmark_tracker.landmark_ids[2], True)
        self.assertEqual(len(self.landmark_tracker.landmark_id_to_feature_index[2]), 3)
        self.landmark_tracker.add_matches(1, 2, np.array([[1, 1]]))
        self.assertEqual(self.landmark_tracker.landmark_ids[2], True)
        self.assertEqual(self.landmark_tracker.landmark_ids[3], True)
        self.assertEqual(len(self.landmark_tracker.landmark_id_to_feature_index[2]), 3)

        self.assertEqual(self.landmark_tracker.landmark_id_to_position[2][0], 1)
        self.assertEqual(self.landmark_tracker.landmark_id_to_position[2][1], 2)
        self.assertEqual(self.landmark_tracker.landmark_id_to_position[2][2], 3)

    def test_add_matches(self) -> None:

        self.assertTrue(self.landmark_tracker.add_features(0, {'keypoints': np.array([[1, 2], [3, 4], [5, 6], [7, 8]])}))
        self.assertTrue(self.landmark_tracker.add_features(1, {'keypoints': np.array([[1, 2], [3, 4], [5, 6], [7, 8]])}))
        self.assertTrue(self.landmark_tracker.add_features(2, {'keypoints': np.array([[1, 2], [3, 4], [5, 6], [7, 8]])}))

        self.landmark_tracker.add_matches(0, 1, np.array([[0, 1], [1, 2], [2, 3]]))
        self.assertEqual(len(self.landmark_tracker.feature_to_landmark_id), 2)
        self.assertEqual(len(self.landmark_tracker.feature_to_landmark_id[0]), 3)

        self.assertEqual(self.landmark_tracker.feature_to_landmark_id[0][0], 1)
        self.assertEqual(self.landmark_tracker.feature_to_landmark_id[0][1], 3)
        self.assertEqual(self.landmark_tracker.feature_to_landmark_id[0][2], 5)

        self.landmark_tracker.assigned_points_3d_if_not_values(0, 0, np.array([0, 1, 2]))
        self.landmark_tracker.assigned_points_3d_if_not_values(0, 1, np.array([1, 2, 3]))
        self.landmark_tracker.assigned_points_3d_if_not_values(0, 2, np.array([2, 3, 4]))

        timestamp_to_camera_index = {0: 0, 1: 1, 2: 2}
        projection_relations, landmark_positions, landmark_id_to_index = self.landmark_tracker.get_projection_relations_and_landmark_position(timestamp_to_camera_index)
        self.assertEqual(len(projection_relations), 6)
        self.assertEqual(len(landmark_positions), 3)
        self.assertEqual(len(landmark_id_to_index), 3)

if __name__ == '__main__':
    unittest.main()