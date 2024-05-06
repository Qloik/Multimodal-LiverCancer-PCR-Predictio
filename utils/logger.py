import os
import sys
import logging

def get_logger(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger_filename = os.path.join(logdir, 'log.txt')
    logging.basicConfig(filename=logger_filename, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S',
                        )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger