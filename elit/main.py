# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:36

import argparse
import sys

from elit import __version__
from elit.server.format import Input


def main():
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    arg_parser = argparse.ArgumentParser(description='ELIT-{}'.format(__version__))
    task_parser = arg_parser.add_subparsers(dest="task", help='which task to perform?')
    parse_parser = task_parser.add_parser(name='parse', help='interactive parse per document')
    server_parser = task_parser.add_parser(name='serve', help='start http server',
                                           description='A http server for ELIT')
    server_parser.add_argument('--port', type=int, default=8000)

    args = arg_parser.parse_args()

    if args.task == 'parse':
        from elit.server.en import en
        for line in sys.stdin:
            line = line.strip()
            doc = en.parse([Input(text=line)])[0]
            print(doc)
    elif args.task == 'serve':
        from elit.server import server
        server.run(port=args.port)


if __name__ == '__main__':
    main()
