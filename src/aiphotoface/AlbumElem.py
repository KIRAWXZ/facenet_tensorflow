

from sets import Set
from FaceElem import FaceElem

class AlbumElem(object):
    _num = "num"
    _average = "average"
    _cover_url = "cover_url"
    _cover_flag = "cover_flag"
    _faces = "faces"
    _tag_tid = "tag_tid"

    # new method
    def new_album_elem(self, num, average, cover_url, cover_flag, face_arr):
        elem = {}
        elem[self._num] = num
        elem[self._average] = average
        elem[self._cover_url] = cover_url
        elem[self._cover_flag] = cover_flag
        elem[self._faces] = face_arr
        return elem


    # set method
    def set_tag_tid(self, album_elem, tag_tid):
        if album_elem != None:
            album_elem[self._tag_tid] = tag_tid
        return album_elem



    # get method
    def get_cover_url(self, album_elem):
        if album_elem != None and album_elem.has_key(self._cover_url):
            return album_elem[self._cover_url]
        return None

    def get_faces(self, album_elem):
        if album_elem != None and album_elem.has_key(self._faces):
            return album_elem[self._faces]
        return None

    def get_fids(self, album_elem):
        faceElem = FaceElem()
        face_arr = self.get_faces(album_elem)
        fids = faceElem.get_fid_arr(face_arr)
        return fids

    def get_fids_set(self, album_elem):
        faceElem = FaceElem()
        face_arr = self.get_faces(album_elem)
        fids = faceElem.get_fid_arr(face_arr)
        return Set(fids)

    def get_tag_tid(self ,album_elem):
        if album_elem != None and album_elem.has_key(self._tag_tid):
            return album_elem[self._tag_tid]
        return None

    def get_num(self, album_elem):
        if album_elem != None and album_elem.has_key(self._num):
            return album_elem[self._num]
        return None

    def get_feature(self, album_elem):
        if album_elem != None and album_elem.has_key(self._average):
            return album_elem[self._average]
        return None


    # update method
    def album_add_face(self, album_elem, face_with_feature, cover_flag, cover_url):
        faceElem = FaceElem()

        # average
        album_elem[self._average] = self._insert_face_change_average(album_elem[self._num], faceElem.get_feature(face_with_feature), album_elem[self._average])

        # num
        album_elem[self._num] += 1
        album_face_num = album_elem[self._num]

        # cover_flag
        if int(album_elem[self._cover_flag]) == 0 and cover_flag == 1:
            album_elem[self._cover_url] = cover_url
            album_elem[self._cover_flag] = 1
        face = faceElem.del_feature(face_with_feature)
        album_elem[self._faces].append(face)
        return album_face_num, album_elem

    def _insert_face_change_average(self, albums_num, feature, albums_average):
        albums_average = [c * albums_num for c in albums_average]
        albums_average = [albums_average[jj] + feature[jj] for jj in range(len(feature))]
        new_average = [c / (albums_num + 1) for c in albums_average]
        return new_average

    def _del_face_change_average(self, albums_num, feature, albums_average):
        albums_average = [c * albums_num for c in albums_average]
        albums_average = [albums_average[jj] - feature[jj] for jj in range(len(feature))]
        new_average = [c / (albums_num - 1) for c in albums_average]
        return new_average

    def del_faces_from_album_elem(self, fid, album_elem, is_solo, feature):
        faceElem = FaceElem()
        face_elems = self.get_faces(album_elem)
        face_elems_new = []
        for face_elem in face_elems:
            if cmp(faceElem.get_fid(face_elem), fid) != 0:
                face_elems_new.append(face_elem)
        album_elem[self._faces] = face_elems_new
        if is_solo:
            album_elem[self._average] = self._del_face_change_average(album_elem[self._num], feature, album_elem[self._average])
            album_elem[self._num] -= 1
        return album_elem

    def add_faces(self, album_elem, faces):
        if album_elem != None and album_elem.has_key(self._faces):
            album_elem[self._faces] = album_elem[self._faces] + faces
        else:
            album_elem[self._faces] = faces
        return album_elem

    def set_faces(self, album_elem, faces):
        album_elem[self._faces] = faces
        return album_elem

    def make_album_cover(self, fileid, faceid):
        return "%s_face_%s.jpg"%(fileid, faceid)