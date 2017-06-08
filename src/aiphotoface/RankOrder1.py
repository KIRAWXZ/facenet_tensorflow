from __future__ import division
import ConfigParser
import numpy as np
import sys
import os
from AlbumElem import AlbumElem
from sets import Set
import pdb


class RankOrder1(object):
    _t = None
    _k = None
    _use_rank_order = False

    def __init__(self):
        # self._k = cf.getint("rankorder1", "rankorder1_k")
        # self._t = cf.getfloat("rankorder1", "rankorder1_t")
        self._k = 18
        self._t = 30


    def absDistance(self, src, dst):
        minv = 1000
        for s in src:
            for d in dst:
                dist = np.linalg.norm(np.mat(self._data[s]) - np.mat(self._data[d]))
                if minv > dist:
                    minv = dist
        return minv

    def genOrderList(self, allset, p ,pidx = None):
        idx = 0
        result = {}
        for v in allset:
            minv = self.absDistance(p, v)
            result[idx] = minv
            idx = idx + 1
        ret = sorted(result.items(), key=lambda d: d[1])
        if pidx:
            if ret[0][0] != pidx:
                for i in range(len(ret)):
                    if ret[i][0] == pidx:
                        ret[0], ret[i] = ret[i], ret[0]
                        break
        return ret

    def roDistance(self, idxsrc, idxdst, allset):
        src = allset[idxsrc]
        dst = allset[idxdst]

        srcol = self.genOrderList(allset, src, idxsrc)
        dstol = self.genOrderList(allset, dst, idxdst)

        rod1 = 0
        roidx1 = 0
        for i in srcol:
            p = 0
            for j in dstol:
                if i[0] == j[0]:
                    break
                p = p + 1
            if p == 0:
                break;
            if roidx1 == 0:
                roidx1 = p
            rod1 = rod1 + p

        rod2 = 0
        roidx2 = 0
        for i in dstol:
            p = 0
            for j in srcol:
                if i[0] == j[0]:
                    break
                p = p + 1
            if p == 0:
                break;
            if roidx2 == 0:
                roidx2 = p
            rod2 = rod2 + p

        if roidx1 == 0 or roidx2 == 0:
            print "qingsong", idxsrc, idxdst,"source:", srcol[:3], "dest", dstol[:3]
            return 3

        return (float(rod1 + rod2)) / float(min(roidx1,roidx2))

    def roNDistance(self, idxsrc, idxdst, allset):
        src = allset[idxsrc]
        dst = allset[idxdst]

        # af = np.concatenate((src,dst))
        af = src + dst

        aset = []
        for i in allset:
            for v in i:
                aset.append([v])

        norm = 0
        for v in af:
            vlist = [v]
            alist = self.genOrderList(aset, vlist)

            n = 0
            for i in range(1, self._k < len(alist) and self._k or len(alist)):
                n = n + (alist[i])[1]
            norm = norm + (n / (self._k < len(alist) and self._k or len(alist)))

        d = self.absDistance(src, dst)

        fai = float(norm) / float(len(af))
        ret = d / fai

        return ret


    def check_merge(self, allset, i ,j):
        distance = self.roDistance(i, j, allset)
        rn = self.roNDistance(i, j, allset)
        self._t = max(len(allset) * 0.2, 4)
        if distance < self._t and rn < 1:
            return True
        return False



    def merge_face(self, allset):
        n = len(allset)
        new_set_num = n

        new_set_index_key = set([])
        for index in range(new_set_num):
            new_set_index_key.add(n - new_set_num + index)

        if n < 2:
            return allset, new_set_index_key
        merge_result_matrix = np.matrix(np.full((n, n), 0))

        index_i = 1
        while index_i < n:

            index_j = 0
            while index_j < index_i:
                if merge_result_matrix[index_i, index_j] == 0:
                    if self.check_merge(allset, index_i, index_j):
                        # merge the set
                        allset[index_j] = allset[index_i] + allset[index_j]
                        del (allset[index_i])

                        # change merge_result_matrixshape = {tuple} <type 'tuple'>: (10, 10)
                        merge_result_matrix[index_j] = 0
                        merge_result_matrix[:, index_j] = 0
                        merge_result_matrix = np.delete(merge_result_matrix, index_i, 0)
                        merge_result_matrix = np.delete(merge_result_matrix, index_i, 1)

                        # reset index_i and index_j
                        n = len(allset)
                        if n < 2:
                            return allset, range(len(allset))

                        index_i = 0
                        index_j = 1
                        print "merge done", len(allset)
                    else:
                        merge_result_matrix[index_i, index_j] = 1

                index_j = index_j + 1
            index_i = index_i + 1

        # return allset, new_set_index_key
        return allset, range(len(allset))

    def deal(self, all_face_list):
        allset = self.ready_set(all_face_list)
        new_all_set = self.merge_face(allset)
        return new_all_set

    def ready_set(self, all_face_list):
        self._data = all_face_list
        all_set = []
        for idx in range(len(all_face_list)):
            all_set.append([idx])
        return all_set