
import json
from FaceElem import FaceElem
from sets import Set
import time

class ClusterMergeElemHttpParam(object):
    _old_album_id = "_old_album_ids"
    _new_album_face_arr = "_new_album_face_arr"
    _faceElem = FaceElem()

    def new_one(self ):
        param = {}
        param[self._old_album_id] = []
        param[self._new_album_face_arr] = []
        return param

    def add_face(self, param, fid, face_id):
        face_elem = self._faceElem.new_face(face_id, fid)
        param[self._new_album_face_arr].append(face_elem)
        return param

    def add_album_id(self, param, album_id):
        param[self._old_album_id].append(album_id)
        return param

    def get_album_ids(self, param):
        if param and param.has_key(self._old_album_id):
            return param[self._old_album_id]
        return None

    def get_face_arr(self, param):
        if param and param.has_key(self._new_album_face_arr):
            return param[self._new_album_face_arr]
        return None

    def unique_list_to_set(self, param):
        param[self._old_album_id] = list(set(param[self._old_album_id]))
        return param


class ClusterMergeHttpParam(object):
    _userid = "_userid"
    _elem_arr = "_elem_arr"
    _unix_time = "_unix_time"
    _fid = "_fid"
    _fileid = "_fileid"

    def new_one(self, userid, fid, fileid):
        params = {}
        params[self._userid] = userid
        params[self._fid] = fid
        params[self._fileid] = fileid
        params[self._elem_arr] = []
        params[self._unix_time] = time.time()
        return params

    def add_cluster_elem(self, params, param_elem):
        params[self._elem_arr].append(param_elem)
        return params

    def get_userid(self, params):
        if params and params.has_key(self._userid):
            return params[self._userid]
        return None

    def get_fid(self, params):
        if params and params.has_key(self._fid):
            return params[self._fid]
        return None

    def get_fileid(self, params):
        if params and params.has_key(self._fileid):
            return params[self._fileid]
        return None

    def get_param_elem_arr(self, params):
        if params and params.has_key(self._elem_arr):
            return params[self._elem_arr]
        return None