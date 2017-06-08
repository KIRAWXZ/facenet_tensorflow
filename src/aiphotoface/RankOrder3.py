from scipy import spatial, array
import sys
import numpy as np
from FDistance import FDistance



# local_thred=0.35
# local_percent=0.06


# local_thred=0.001
# local_percent=0.11

local_thred=0.0000000150
local_percent=0.01

class RankOrder3(object):

    _tree = None
    _data = None
    _rank_order_list = None

    _ready_set_thred=local_thred
    _percent=local_percent
    _qiebixuefu_thred = 0.035
    _fdistance = FDistance()


    def _make_tree(self, all_face_list):
        _tree = spatial.KDTree(all_face_list)
        self._data = all_face_list
        self._rank_order_list = {}
        face_number = len(all_face_list)
        for idx, face in enumerate(all_face_list):
            order_list = self._order_list(_tree, idx, face_number, self._ready_set_thred)
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

    def _order_list(self, _tree, index, face_number, distance_upper_bound = None):
        node = self._data[index]
        order_list = _tree.query(x=node, p=2, k=face_number)
        index_thred = len(order_list[0])
        if distance_upper_bound != None:
            dis_order_list = list(order_list[0])
            for idx, dis in enumerate(dis_order_list):
                if dis**2 > distance_upper_bound:
                    index_thred = idx
                    break
        index_order_list = list(order_list[1])
        return index_order_list[:index_thred]


    def check_merge(self, aset, bset):
        common_count = len(list(set(aset).intersection(set(bset))))
        sum_count = len(list(set(aset).union(set(bset))))
        return float(common_count)/float(sum_count) > self._percent

    # def check_merge(self, alist, blist):
    #     a_percent, p_percent, intersection  = self.get_inter_weight_percent(alist, blist)
    #     return (a_percent + p_percent) * 0.5 > self._percent

    def get_inter_weight_percent(self, alist, blist):
        # if alist:
        #     return 0, 1, []
        # elif blist:
        #     return 1, 0, []
        aset = set(alist)
        bset = set(blist)
        intersection = list(aset.intersection(bset))
        alen = len(aset)
        blen = len(bset)
        a_index = 0
        b_index = 0
        if alen == 0:
            print "bad"
        for idx in intersection:
            a_index = a_index + alen - alist.index(idx)
            b_index = b_index + blen - blist.index(idx)
        a_percent = float(a_index) / float((alen+1)*alen/2)
        b_percent = float(b_index) / float((blen+1)*blen/2)
        return a_percent, b_percent, intersection




    def ready_set(self):
        all_set = []
        for idx, one_set in self._rank_order_list.items():
            all_set.append(one_set)
        return all_set


    def merge_operation(self, alist, blist):
        clist = list(set(alist).union(set(blist)))
        face_map = []
        face_list = []
        for s_idx in clist:
            face_map.append(s_idx)
            face_list.append(self._data[s_idx])
        _tree = spatial.KDTree(face_list)
        f_average = np.mean(face_list, 0)
        sorted_index = _tree.query(x=f_average, p=2, k=len(clist))
        s_index = [face_map[idx] for idx in sorted_index[1]]
        return s_index



    def merge_face(self):
        allset = self.ready_set()

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
                    merge = self.check_merge(allset[index_i], allset[index_j])
                    if merge:
                        # merge the set
                        allset[index_j] = self.merge_operation(allset[index_i] , allset[index_j])
                        # allset[index_j] = list(set(allset[index_i] + allset[index_j]))
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
        self._make_tree(all_face_list)
        new_all_set, num = self.merge_face()
        # new_all_set = self.remove_duplicate(new_all_set)
        return new_all_set

    def remove_duplicate(self, allset):
        i = 0
        empty_index = set([])
        while i < len(allset):
            j = 0
            while j < i:
                a_percent, b_percent ,intersection = self.get_inter_weight_percent(allset[i], allset[j])
                if a_percent < b_percent:
                    allset[i] = np.delete(np.array(allset[i]), [allset[i].index(k) for k in intersection if k in allset[i]]).tolist()
                    if len(allset[i]) == 0:
                        empty_index.add(i)
                else:
                    allset[j] = np.delete(np.array(allset[j]), [allset[j].index(k) for k in intersection if k in allset[j]]).tolist()
                    if len(allset[j]) == 0:
                        empty_index.add(j)
                j = j + 1
            i = i + 1
        new_set = np.delete(np.array(allset), list(empty_index)).tolist()
        return new_set




