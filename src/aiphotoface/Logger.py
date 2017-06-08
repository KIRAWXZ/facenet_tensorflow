#!/usr/bin/env python
#coding: utf-8

import logging, logging.handlers
import sys
import traceback
import time
import os

class Logger(object):
    logger = None

    def __init__(self, _path, _app, _level = logging.INFO, _format='%(asctime)s %(levelname)s %(message)s'):
        formatter = logging.Formatter(_format)
        self._create_path(_path)
        hdlr = logging.handlers.TimedRotatingFileHandler(_path + "/" + _app + "_" + str(os.getpid()), 'midnight', 1, 3)
        hdlr.suffix = "%Y%m%d_%H%M%S.log"
        hdlr.setFormatter(formatter)
        self.logger = logging.getLogger()
        self.logger.setLevel(_level)
        self.logger.addHandler(hdlr) 


    def log_war(self,_msg):
        self.logger.warning(_msg)

    def log_inf(self, _msg):
        self.logger.info(_msg)

    def log_err(self, _msg):
        self.logger.error(_msg)

    def log_exc(self, _msg):
        self.logger.exception(_msg)

    def _create_path(self, _path):
        if not os.path.exists(_path):
            os.makedirs(_path)

