

from FeatureElem import FeatureElem
import json

class UserFeature(object):
    _album_redis = None
    _userid_fids_suffix = None

    def __init__(self, cf, album_redis):
        self._userid_fids_suffix = cf.get("user", "user_features_suffix")
        self._album_redis = album_redis

    def get_userid_fids_key(self, userid):
        return self._userid_fids_suffix%(userid)

    def ready_userid_all_fids_features(self, userid):
        key_str = self.get_userid_fids_key(userid)
        return self._album_redis.hgetall(key_str)

    def set_userid_all_fids_features(self, userid, features):
        key_str = self.get_userid_fids_key(userid)
        self._album_redis.delete(key_str)
        return self._album_redis.hmset(key_str, features)

    def get_feature_by_userid_fid(self, userid, fid):
        key_str = self.get_userid_fids_key(userid)
        return self._album_redis.hget(key_str, fid)

    def get_feature_by_userid_fid_and_faceid(self, userid, fid, faceid):
        feature_str = self.get_feature_by_userid_fid(userid, fid)
        featureElem = FeatureElem()
        return featureElem.get_feature_from_feature_elem_by_faceid(json.loads(feature_str), faceid)

    def get_fileid_by_userid_fid(self, userid, fid):
        feature_str = self.get_feature_by_userid_fid(userid, fid)
        feature_obj = json.loads(feature_str)
        featureElem = FeatureElem()
        return featureElem.get_fileid(feature_obj)