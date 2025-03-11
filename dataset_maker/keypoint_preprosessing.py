import numpy as np

class keypoint_preprosessing:
    def __init__(self, norm_type = None):
        self.norm_type = norm_type
        
    def _points_normalization(self, keypoint_matrix):
        matrix_norm = np.linalg.norm(keypoint_matrix)
        keypoint_matrix = keypoint_matrix/(matrix_norm + 0.0001)
        
        return keypoint_matrix
        
    def _coordinate_convert(self, keypoints):
        converted_keypoints = []
        keypoints = np.array(keypoints)
        
        [x1, x2] = [keypoints[13], keypoints[12]]
        [y1, y2] = [keypoints[1], keypoints[0]]
        
        s_vector = (y1 - x1) + (y2 - x2)
        
        x_vector = (x2 - x1)/np.linalg.norm(x2 - x1)
        z_vector = np.cross(x_vector, s_vector)
        z_vector /= np.linalg.norm(z_vector)
        
        y_vector = np.cross(x_vector, z_vector)
        y_vector /= np.linalg.norm(y_vector)
        
        rotation_matrix = np.column_stack((x_vector, y_vector, z_vector))
        middle_point = ((x1 + x2)/2) @ rotation_matrix

        converted_keypoints = [(tuple((kp @ rotation_matrix) - middle_point)) for kp in keypoints]

        return converted_keypoints

    def prosess(self, keypoints):
        converted_keypoints = self._coordinate_convert(keypoints)

        if self.norm_type != None:
            preprosessed_keypoints = [tuple(kp) for kp in self._points_normalization(np.array(converted_keypoints))]
        else:
            preprosessed_keypoints = converted_keypoints
            
        return preprosessed_keypoints