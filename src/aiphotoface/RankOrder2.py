import numpy as np
from scipy import spatial, array
import sys
import multiprocessing
import itertools as it
import numpy as np
import copy_reg
import time
import types
# import pp

# def _pickle_method(m):
#     if m.im_self is None:
#         return getattr, (m.im_class, m.im_func.func_name)
#     else:
#         return getattr, (m.im_self, m.im_func.func_name)
#
# copy_reg.pickle(types.MethodType, _pickle_method)



class RankOrder2(object):

    _tree = None
    _data = None
    _rank_order_list = None
    _dm_save = {}

    _k = 10
    _thred = 15
    _percent = 0.8
    _DM_save = {}

    # def ready_more_cpu_pool(self):
    #     # cores = multiprocessing.cpu_count()
    #     cores = 10
    #     pool = multiprocessing.Pool(processes=cores)
    #     return pool

    # def ready_pp(self):
    #     ppservers = ()
    #     job_server = pp.Server(ppservers=ppservers)
    #     return job_server

    # pool
    # def ready_DM(self):
    #     pool = self.ready_more_cpu_pool()
    #     xs = range(len(self._data))
    #     tuple_list = list(it.combinations(xs, 2))
    #     tuple_list_param = []
    #     for idx, tuple in enumerate(tuple_list):
    #         tuple_list_param.append((tuple[0], tuple[1], idx))
    #     # DM_list = pool.map(self._Dm_pool_param, tuple_list)
    #     DM_list = pool.map(self._Dm_pool_param, tuple_list_param)
    #     for idx, DM in enumerate(DM_list):
    #         tuple = tuple_list[idx]
    #         key = str(min(tuple[0], tuple[1])) +"_" + str(max(tuple[0], tuple[1]))
    #         self._DM_save[key] = DM

    # pp
    # def ready_DM(self):
    #     job_server = self.ready_pp()
    #     xs = range(len(self._data))
    #     tuple_list = list(it.combinations(xs, 2))
    #     # tuple_list_param = []
    #     # for tuple in tuple_list:
    #     #     aorderlist = self._rank_order_list[tuple[0]]
    #     #     borderlist = self._rank_order_list[tuple[1]]
    #     #     tuple_list_param.append((tuple[0], tuple[1], aorderlist, borderlist))
    #     print "task %d" % len(tuple_list)
    #     jobs = [(idx, job_server.submit(self._Dm_pool_param, (tuple,), (self._Dm_get, self._dm, self._O_index,))) for idx, tuple in enumerate(tuple_list)]
    #     print "commit done"
    #     for idx, job in jobs:
    #         tuple = tuple_list[idx]
    #         key = str(min(tuple[0], tuple[1])) +"_" + str(max(tuple[0], tuple[1]))
    #         self._DM_save[key] = job()
    #         print 'ready-DM', tuple, self._DM_save[key]

    #single thread
    def ready_DM(self):
        # pool = self.ready_more_cpu_pool()
        xs = range(len(self._data))
        tuple_list = list(it.combinations(xs, 2))
        # DM_list = pool.map(self._Dm_pool_param, tuple_list)
        for idx, tuple in enumerate(tuple_list):
            value = self._Dm_pool_param(tuple)
            key = str(min(tuple[0], tuple[1])) +"_" + str(max(tuple[0], tuple[1]))
            self._DM_save[key] = value

    def _Dm_pool_param(self, tuple):
        result = self._Dm_get(tuple[0], tuple[1])
        return result

    def _make_tree(self, all_face_list):
        # all_face_tuple = []
        # for face in all_face_list:
        #     all_face_tuple.append(face)
        _tree = spatial.KDTree(all_face_list)
        self._data = all_face_list
        self._rank_order_list = {}
        face_number = len(all_face_list)
        for idx, face in enumerate(all_face_list):
            order_list = self._order_list(_tree, idx, face_number)
            if order_list[0] != idx:
                order_list[1], order_list[0] = order_list[0], order_list[1]
            self._rank_order_list[idx] = order_list

    def _O_index(self, index_a, index_b):
        alist = self._rank_order_list[index_a]
        try:
            rank = alist.index(index_b)
        except:
            rank = sys.maxint
        return rank

    def _order_list(self, _tree, index, number):
        node = self._data[index]
        order_list = _tree.query(node, number)
        index_order_list = []
        for index in order_list[1]:
            index_order_list.append(index)
        return index_order_list

    # def _dm(self, index_a, index_b):
    #     alist = self._rank_order_list[index_a]
    #     min_count = min(self._k, self._O_index(index_a, index_b))
    #     result_distance = 0
    #     for alist_index in alist[:min_count]:
    #         b_index = self._O_index(index_b, alist_index)
    #         if b_index > self._k:
    #             result_distance = result_distance + 1
    #     return result_distance

    def _dm(self, index_a, index_b):
        key = str(index_a) + "_" + str(index_b)
        if self._dm_save.has_key(key):
            result = self._dm_save[key]
        else:
            srcol = self._rank_order_list[index_a]
            dstol = self._rank_order_list[index_b]
            rod1 = 0
            roidx1 = 0
            for i in srcol:
                p = dstol.index(i)
                if p == 0:
                    break
                if roidx1 == 0:
                    roidx1 = p
                rod1 = rod1 + p
            result = rod1
        return result



    def _Dm_get(self, index_a, index_b):
        # key = str(min(index_a, index_b)) + "_" + str(max(index_a, index_b))
        # if self._DM_save.has_key(key):
        #     return self._DM_save[key]
        dm_ab = self._dm(index_a, index_b)
        dm_ba = self._dm(index_b, index_a)
        Oa_b = self._O_index(index_a, index_b)
        Ob_a = self._O_index(index_b, index_a)

        if Ob_a == 0 or Oa_b == 0:
            print "bad"

        result = float(dm_ab + dm_ba) / float(max( min(Ob_a, Oa_b), 1))
        # result = float(dm_ab + dm_ba) / float(self._k*2)
        # print "Dm", dm_ba, dm_ba, Oa_b, Ob_a, result
        # self._DM_save[key] = result
        return result

    def _Dm(self, index_a, index_b):
        key = str(min(index_a, index_b)) + "_" + str(max(index_a, index_b))
        return self._DM_save[key]

    def check_merge(self, aset, bset, n):
        alen = len(aset)
        blen = len(bset)
        merge_count = 0
        merge_count_thred = alen*blen*self._percent
        # self._thred = max(n * 0.3, 4)
        for aface_index in aset:
            for bface_index in bset:
                dm = self._Dm(aface_index, bface_index)
                print "dm", dm
                if dm <= self._thred and min(self._O_index(aface_index, bface_index), self._O_index(bface_index, aface_index)):
                    merge_count = merge_count+1
                    if merge_count >= merge_count_thred:
                        return True
        # print "percent", float(merge_count)/float(alen*blen)
        return merge_count >= merge_count_thred

    def ready_set(self, all_face_list):
        all_face_set = []
        for index in range(len(all_face_list)):
            all_face_set.append([index])
        return all_face_set


    def merge_face(self, all_face_list):
        allset = self.ready_set(all_face_list)

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
                    if self.check_merge(allset[index_i], allset[index_j], n):
                        # merge the set
                        allset[index_j] = allset[index_i] + allset[index_j]
                        del (allset[index_i])

                        # change the new_set_index
                        # new_set_index_key.discard(index_i)
                        # new_set_index_key.add(index_j)
                        # for index in new_set_index_key:
                        #     if index > index_i:
                        #         new_set_index_key.discard(index)
                        #         new_set_index_key.add(index - 1)

                        # change merge_result_matrixshape = {tuple} <type 'tuple'>: (10, 10)
                        merge_result_matrix[index_j] = 0
                        merge_result_matrix[:, index_j] = 0
                        merge_result_matrix = np.delete(merge_result_matrix, index_i, 0)
                        merge_result_matrix = np.delete(merge_result_matrix, index_i, 1)

                        # reset index_i and index_j
                        n = len(allset)
                        if n < 2:
                            # return allset, new_set_index_key
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
        # print "bofore", time.time()
        self._make_tree(all_face_list)
        # print "make_tree", time.time()
        self.ready_DM()
        # print "ready dm", time.time()
        new_all_set, num = self.merge_face(all_face_list)
        # print "merge set", time.time()
        return new_all_set