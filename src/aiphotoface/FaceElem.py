
class FaceElem(object):
    _face_id = "face_id"
    _fid = "fid"
    _feature ="feature"


    # new method
    def new_face(self, faceid, fid):
        face = {}
        face[self._face_id] = faceid
        face[self._fid] = fid
        return face

    def new_face_with_feature(self, faceid, fid, feature):
        face = self.new_face(faceid,fid)
        face[self._feature] = feature
        return face

    def new_face_with_feature2(self, face, feature):
        face[self._feature] = feature
        return face

    # get method
    def get_feature(self, face):
        if face != None and face.has_key(self._feature):
            return face[self._feature]
        return None

    def get_fid(self, face):
        if face != None and face.has_key(self._fid):
            return face[self._fid]
        return None

    def get_fid_arr(self, face_arr):
        fids = []
        for face in face_arr:
            if self.get_fid(face):
                fids.append(self.get_fid(face))
        return fids

    def get_face_id(self, face):
        if face != None and face.has_key(self._face_id):
            return face[self._face_id]
        return None



    # delete method
    def del_feature(self, face):
        if face != None and face.has_key(self._feature):
            del face[self._feature]
        return face

class FaceDict(FaceElem):
    _fileid = "fileid"

    def new_dict(self, fid, fileid, faceid):
        face = self.new_face(faceid, fid)
        face[self._fileid] = fileid
        return face

    def get_fileid(self, face):
        if face and face.has_key(self._fileid):
            return face[self._fileid]
        return None

    def to_elem(self, face):
        fid = self.get_fid(face)
        faceid = self.get_face_id(face)
        return self.new_face(faceid, fid)