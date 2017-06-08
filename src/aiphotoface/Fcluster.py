# -*- coding: utf-8 -*-
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


class Fcluster:

    def beginClusters(self, all_face_list):
        thed = 0.0001224
        cluster_tree = linkage(all_face_list, method = 'average', metric = 'euclidean')
        return fcluster(cluster_tree, thed, criterion='distance')

    def ready_set(self, clusters):
        all_set = {}
        for idx, set_id in enumerate(clusters):
            if all_set.has_key(set_id):
                all_set[set_id].append(idx)
            else:
                all_set[set_id] = [idx]
        return all_set.values()

    def deal(self, all_face_list):
        clusters = self.beginClusters(all_face_list)
        return self.ready_set(clusters)



