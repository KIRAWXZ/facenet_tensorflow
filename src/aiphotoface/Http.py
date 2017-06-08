#!/usr/bin/env python
#coding=utf8

import requests
import traceback


class Http(object):
    _log = None
    _ip = None
    _port = None
    _host = "http://%s:%d%s"
    _timeout = 10

    def __init__(self, log, ip = None , port = None):
        self._log = log
        self._ip = ip
        self._port = port


    def post_urn(self, urn, d, header = None):
        if not self._ip or not self._port:
            self._log.log_err("ip(%s) or port(%s) none"%(self._ip, self._port))
            return None
        try:
            r = requests.post(self._host%(self._ip, self._port, urn), data = d, headers = header)
            return r.text
        except:
            self._log.log_exc("%s, %s , %s, %s"%(self._ip, self._port, urn, d))
        return None

    def delete_urn(self, urn, d, header = None):
        if not self._ip or not self._port:
            self._log.log_err("ip(%s) or port(%s) none"%(self._ip, self._port))
            return None
        try:
            r = requests.delete(self._host%(self._ip, self._port, urn), data = d, headers = header)
            return r.text
        except:
            self._log.log_exc("%s, %s , %s, %s"%(self._ip, self._port, urn, d))
        return None

    def post_url(self, url, data = None, headers = None, params = None):
        try:
            r = requests.post(url, data = data, headers = headers, params = params)
            return r.text
        except:
            self._log.log_exc("%s, %s, %s, %s"%(url, data, headers, params))
        return None

    def get_url(self, url, data = None, headers = None):
        try:
            r = requests.get(url, data = data, headers = headers)
            return r.text
        except:
            self._log.log_exc("%s, %s, %s"%(url, data, headers))
        return None

    def del_url(self, url, data, headers = None):
        try:
            r = requests.delete(url, data = data, headers = headers)
            return r.text
        except:
            self._log.log_exc("%s, %s, %s"%(url, data, headers))
        return None
