
import numpy as np
import scipy.spatial.distance as dist

class FDistance(object):
    def oushi_distance(self, f1, f2):
        diff_temp = np.array(f1) - np.array(f2)
        dist = np.linalg.norm(diff_temp)
        return dist

    def cos_distance(self, f1, f2):
        up = np.matrix(f1)*np.matrix(f2).T

        down = np.linalg.norm(f1) * np.linalg.norm(f2)
        return up/down

    def manhadun_distance(self, f1, f2):
        sum_d = 0
        for idx in range(len(f1)):
            sum_d = sum_d + abs(f1[idx] - f2[idx])
        return sum_d

    def qiebixuefu_distance(self, f1, f2):
        max_d = 0
        for idx in range(len(f1)):
            max_d = max(max_d , abs(f1[idx] - f2[idx]))
        return max_d


    def jaccard_Similarity_Coefficient(self, f1, f2):
        matV = np.mat([f1,f2])
        return dist.pdist(matV,'jaccard')