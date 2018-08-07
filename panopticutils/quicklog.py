import time
import datetime
import logging

class QuickLogger:
    
    def __init__(self, file):
        self.logger = logging.getLogger()
        self.setup_file_logger(file)
        return
    
    def setup_file_logger(self, log_file):
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr) 
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        self.logger.info(message)