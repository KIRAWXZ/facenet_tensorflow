

import json


class RankOrderKeyElem(object):
    _new_flag="new_flag"
    _album_id="album_id"
    _face_id="face_id"
    _fid = "fid"

    def new_elem(self, new_flag, album_id, face_id, fid):
        key_obj = {}
        key_obj[self._new_flag] = new_flag
        key_obj[self._album_id] = album_id
        key_obj[self._face_id] = face_id
        key_obj[self._fid] = fid
        return key_obj

    def is_new(self, key_obj):
        if key_obj and key_obj.has_key(self._new_flag):
            return key_obj[self._new_flag]
        return False

    def get_album_id(self, key_obj):
        if key_obj and key_obj.has_key(self._album_id):
            return key_obj[self._album_id]
        return None

    def get_face_id(self, key_obj):
        if key_obj and key_obj.has_key(self._face_id):
            return key_obj[self._face_id]
        return None

    def get_fid(self, key_obj):
        if key_obj and key_obj.has_key(self._fid):
            return key_obj[self._fid]
        return None