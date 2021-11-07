import logging
import logging.config
import parser
import datetime
import os
import sys

from cifar import create_classIL_task
from parse import set_parser

logger = logging.getLogger(__name__)

def main(args):

    t = datetime.datetime.now()
    timestamp = '%0*i%0*i%0*i%0*i%0*i%0*i' % (4, t.year, 2, t.month, 2, t.day, 2, t.hour, 2, t.minute, 2, t.second)
    args.timestamp = timestamp
    save_dir = os.path.join(args.save_root, 'cifar100_B%i_%isteps_%s' % (args.base_task_cls, args.steps, args.timestamp))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    log_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'form01': {
                'format': "%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                'datefmt': "%m/%d/%Y %H:%M:%S",
            }
        },
        'handlers': {
            'ch': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'form01',
                'stream': 'ext://sys.stderr'
            },
            'fh': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'form01',
                'filename': os.path.join(save_dir, 'data_generation.log'),
                'mode': 'w+'
            }
        },
        'root': {
            'handlers': ['ch', 'fh'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(log_dict)
    logger.info(dict(args._get_kwargs()))
    create_classIL_task(args)

if __name__ == '__main__':

    args = set_parser()
    main(args)