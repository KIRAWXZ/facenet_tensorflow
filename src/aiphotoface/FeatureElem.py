
import json

class Face(object):
    _face_id = "face_id"
    _feature = "feature"

    def get_face_id(self, face):
        if face != None and face.has_key(self._face_id):
            return face[self._face_id]
        return None

    def get_feature(self, face):
        if face != None and face.has_key(self._feature):
            return face[self._feature]
        return None

class FeatureElem(object):
    _fileid = "fileid"
    _features = "features"
    _face = Face()

    def get_fileid(self, feature_elem):
        if feature_elem != None and feature_elem.has_key(self._fileid):
            return feature_elem[self._fileid]
        return None

    def get_feature_arr(self, feature_elem):
        if feature_elem != None and feature_elem.has_key(self._features):
            return feature_elem[self._features]
        return None

    def is_solo_face(self, feature_elem):
        feature_arr = self.get_feature_arr(feature_elem)
        if feature_arr != None and len(feature_arr) == 1:
            return True, feature_arr[0]
        return False, None

    def get_feature_from_feature_face(self, feature_face):
        return self._face.get_feature(feature_face)

    def get_face_id_from_feature_face(self, feature_face):
        return self._face.get_face_id(feature_face)

    def get_feature_from_feature_elem_by_faceid(self, feature_elem, faceid):
        feature_arr = self.get_feature_arr(feature_elem)
        for face in feature_arr:
            if cmp(self._face.get_face_id(face), faceid) == 0:
                return self._face.get_feature(face)
        return None







