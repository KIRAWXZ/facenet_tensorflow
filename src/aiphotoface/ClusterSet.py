#!/usr/bin/env python
#coding=utf-8

import json
import os

import numpy as np
from AlbumElem import AlbumElem
from ClusterMergeParam import ClusterMergeElemHttpParam
from ClusterMergeParam import ClusterMergeHttpParam
from FaceElem import FaceElem,FaceDict
from FeatureElem import FeatureElem
from RankOrderKeyElem import RankOrderKeyElem

from RankOrder1 import RankOrder1
from RankOrder2 import RankOrder2
from RankOrder3 import RankOrder3
from Fcluster import Fcluster

class ClusterSet(object):
    _rank_order = None
    _album_redis = None
    _g_UserFeature = None
    _g_http = None
    _logger = None
    _featureElem = None
    _ClusterMergeHttpParam = None
    _ClusterMergeElemHttpParam = None
    _rankOrderKeyElem = None
    _api_cluster_merge_set = None

    def __init__(self, logger, cf, album_redis, g_UserFeature, g_http):
        self._fcluster = Fcluster()
        self._rank_order1 = RankOrder1()
        self._rank_order2 = RankOrder2()
        self._rank_order3 = RankOrder3()
        self._album_redis = album_redis
        self._g_UserFeature = g_UserFeature
        self._g_http = g_http
        self._logger = logger
        self._featureElem = FeatureElem()
        self._ClusterMergeHttpParam = ClusterMergeHttpParam()
        self._ClusterMergeElemHttpParam = ClusterMergeElemHttpParam()
        self._rankOrderKeyElem = RankOrderKeyElem()
        self._api_cluster_merge_set = cf.get("machine", "api_cluster_set")
        self._data_path = cf.get("cluster", "cluster_data_path")


###############################
# cluster module for add one face
###############################
    def use_cluster_set(self):
        return True


    def change_albums_to_cluster_set(self, albums , userid):
        allset = []
        albumElem = AlbumElem()
        faceElem = FaceElem()
        if albums:
            for album_key in albums.keys():
                album_elem = json.loads(albums[album_key])
                one_cluster_set = []
                for face in albumElem.get_faces(album_elem):
                    feature_elem_str = self._userid_features[faceElem.get_fid(face)]
                    feature_elem = json.loads(feature_elem_str)
                    feature = self._featureElem.get_feature_from_feature_elem_by_faceid(feature_elem, faceElem.get_face_id(face))
                    if feature:
                        one_cluster_set.append(self.new_one_cluster_face_elem(False, album_key, None, feature, None))
                    else:
                        self._logger.logger.log_err(
                            "method[%s] userid[%s] album_id[%s] face[%s] msg[%s]" % ("change_album_to_cluster_set", userid, album_key, json.dumps(face), "get feature error"))
                allset.append(one_cluster_set)
        return allset

    def new_one_cluster_face_elem(self, is_new, album_id, face_id, feature, fid):
        key_obj = self._rankOrderKeyElem.new_elem(is_new, album_id, face_id, fid)
        key_str = json.dumps(key_obj)
        return (key_str, np.matrix(feature))

    def new_cluster_set_with_fid(self, fid):
        all_set = []
        # get all the face with the fid
        feature_json = self._userid_features[fid]
        feature_elem = json.loads(feature_json)
        feature_arr = self._featureElem.get_feature_arr(feature_elem)
        for feature_face in feature_arr:
            face_id = self._featureElem.get_face_id_from_feature_face(feature_face)
            feature = self._featureElem.get_feature_from_feature_elem_by_faceid(feature_elem, face_id)
            all_set.append([self.new_one_cluster_face_elem(True, None, face_id, feature, fid)])
        return all_set

    def ready_allset_for_rank_order(self, old_albums, fid_arr, userid):
        all_set1 = self.change_albums_to_cluster_set(old_albums, userid)
        all_set2_arr = []
        for fid in fid_arr:
            all_set2 = self.new_cluster_set_with_fid(fid)
            all_set2_arr = all_set2_arr + all_set2
        return all_set1 + all_set2_arr, len(all_set2_arr)

    def ready_params_for_merge_machine_albums(self, all_cluster_sets, new_set_indexs, userid, fid, fileid):
        params = self._ClusterMergeHttpParam.new_one(userid, fid , fileid)
        for new_set_index in new_set_indexs:
            new_cluster_set = all_cluster_sets[new_set_index]
            param_elem = self._ClusterMergeElemHttpParam.new_one()
            for rank_order_face in new_cluster_set:
                key_obj = json.loads(rank_order_face[0])
                if self._rankOrderKeyElem.is_new(key_obj):
                    face_id = self._rankOrderKeyElem.get_face_id(key_obj)
                    fid = self._rankOrderKeyElem.get_fid(key_obj)
                    self._ClusterMergeElemHttpParam.add_face(param_elem, fid, face_id)
                else:
                    album_id = self._rankOrderKeyElem.get_album_id(key_obj)
                    self._ClusterMergeElemHttpParam.add_album_id(param_elem, album_id)
            param_elem = self._ClusterMergeElemHttpParam.unique_list_to_set(param_elem)
            self._ClusterMergeHttpParam.add_cluster_elem(params, param_elem)
        return params

    def post_machine(self, params):
        self._g_http.post_urn(self._api_cluster_merge_set, {"param": json.dumps(params)})

    def deal_for_cluster(self, userid, fid, fileid):
        # get all the features
        self._userid_features = self._g_UserFeature.ready_userid_all_fids_features(userid)

        # get the user albums
        old_albums = self._album_redis.hgetall(userid)
        # print "old_albums", fid, old_albums
        allset, new_number = self.ready_allset_for_rank_order(old_albums, [fid], userid)
        if new_number > 0:
            # print "ready all set",fid, len(allset), new_number
            new_allset, new_set_indexs =  self._rank_order.merge_all_cluster_set(allset, new_number)
            # print "merge done", fid, len(new_allset), new_set_indexs
            param_obj = self.ready_params_for_merge_machine_albums(new_allset, new_set_indexs, userid, fid, fileid)
            # print "ready param done", fid, param_obj
            result = self.post_machine(param_obj)
            self._logger.logger.log_war(
                "method[%s] userid[%s] fid[%s] msg[machine_response: %s]" % (
                "deal", userid, fid, result))
        else:
            self._logger.logger.log_err(
                "method[%s] userid[%s] fid[%s] msg[%s]" % (
                "deal", userid, fid, "the fid has no face feature"))




    # def re_rankorder_cluster_by_userid(self, userid):
    #     status = False
    #     result_all_set = None
    #     result_new_set_indexs = None
    #     self._userid_features = self._g_UserFeature.ready_userid_all_fids_features(userid)
    #     fid_arr = self._userid_features.keys()
    #     allset, new_number = self.ready_allset_for_rank_order(None, fid_arr, userid)
    #     if new_number > 0:
    #         new_allset, new_set_indexs =  self._rank_order.merge_all_cluster_set_with_new_face(allset, new_number)
    #         new_set_indexs = range(len(new_allset))
    #         status = True
    #         result_all_set = new_allset
    #         result_new_set_indexs = new_set_indexs
    #     else:
    #         self._logger.logger.log_err(
    #             "method[%s] userid[%s] msg[%s]" % (
    #             "re_cluster_by_userid", userid, "new_number = 0"))
    #     return status, result_all_set, result_new_set_indexs



##############################
# 测试特征值
##############################
    def file_name_get_fid_false(self, file_name_list):
        fid_list = []
        for file_name in file_name_list:
            name = os.path.basename(file_name)
            name_arr = name.split(".")
            fid_list.append(name_arr[0])
        return fid_list

    def file_name_get_fid_arr(self, file_name_list):
        fid_list = []
        # print file_name_list
        for file_name in file_name_list:
            name = self.file_name_get_fid_false(file_name)
            fid_list.append(name)
        return fid_list


    def ready_dict_and_list_for_userid_false(self, userid):
        all_face_dict = []
        all_face_list = []

        # fid_fileid_map = self.map_fid_fileid(userid, g_http)

        faceDict = FaceDict()
        file_name_list = np.load("/letv/aiphoto/photo/%s_rst/gallery.npy"%userid)
        fid_list = self.file_name_get_fid_arr(file_name_list)
        features = np.load("/letv/aiphoto/photo/%s_rst/signatures.npy"%userid)
        faceid_list = np.load("/letv/aiphoto/photo/%s_rst/position.npy"%userid)

        # min_count = 100*1
        min_count = len(features)

        for idx, feature in enumerate(features):
            fids = fid_list[idx]
            one_list = []
            for fid in fids:
                fileid = fid
                face_dict = faceDict.new_dict(fid, fileid, fid)
                one_list.append(face_dict)
            all_face_dict.append(one_list)
            all_face_list.append(feature)
        return all_face_dict[:min_count], all_face_list[:min_count], faceid_list


    def ready_dict_and_list_for_userid_from_data_file(self, userid, g_http):
        all_face_dict = []
        all_face_list = []

        fid_fileid_map = self.map_fid_fileid(userid, g_http)
        base_path = self._data_path % userid

        faceDict = FaceDict()
        file_name_list = np.load("%s/gallery.npy"%base_path)
        fid_list = self.file_name_get_fid_arr(file_name_list)
        # fid_list = file_name_list
        features = np.load("%s/signatures.npy"%base_path)
        faceid_list = np.load("%s/position.npy"%base_path)

        for idx, feature in enumerate(features):
            fids = fid_list[idx]
            one_list = []
            for fid in fids:

                # if cmp(fid, "161207105312000019124164210004") == 0:
                #     print faceid_list[idx]

                fileid = fid_fileid_map[fid]
                face_dict = faceDict.new_dict(fid, fileid, fid)
                one_list.append(face_dict)
            all_face_dict.append(one_list)
            all_face_list.append(feature)
        return all_face_dict, all_face_list, faceid_list

###########################################
# cluster for machine module
###########################################

    def map_fid_fileid(self, userid, g_http):
        result_map = {}
        ok, all_fid_fileid = g_http.proxy_get_all_fids_from_uid_new(userid, 300, True)
        if ok:
            for index in range(0, len(all_fid_fileid)):
                fid_fileid = all_fid_fileid[index]
                fid = fid_fileid['fid']
                fileid = fid_fileid['fsha1']
                result_map[fid] = fileid
        return result_map


    def ready_dict_and_list_for_userid(self, user_features):
        all_face_dict = []
        all_face_list = []
        featureElem = FeatureElem()
        faceElem = FaceElem()
        faceDict = FaceDict()
        for fid, v in user_features.items():
            feature_elem = json.loads(v)
            fileid = featureElem.get_fileid(feature_elem)
            for face in featureElem.get_feature_arr(feature_elem):
                faceid = faceElem.get_face_id(face)
                feature = faceElem.get_feature(face)
                face_dict = faceDict.new_dict(fid, fileid, faceid)
                all_face_dict.append(face_dict)
                all_face_list.append(feature)
        return all_face_dict, all_face_list


    def re_cluster_by_userid(self, userid, g_http):
        status = False
        result_all_set = None
        self._userid_features = self._g_UserFeature.ready_userid_all_fids_features(userid)
        all_face_dict, all_face_list, faceid_list = self.ready_dict_and_list_for_userid_from_data_file(userid, g_http)
        # all_face_dict, all_face_list, faceid_list = self.ready_dict_and_list_for_userid_false(userid)

        # all_face_dict, all_face_list = self.ready_dict_and_list_for_userid(self._userid_features)
        if len(all_face_list) > 0:
            result_all_set = self._rank_order3.deal(all_face_list)
            status = True
        else:
            self._logger.logger.log_err(
                "method[%s] userid[%s] msg[%s]" % (
                "re_cluster_by_userid", userid, "new_number = 0"))
        return status, result_all_set, all_face_dict, faceid_list
        # return status, result_all_set, all_face_dict


