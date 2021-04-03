# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:36

import argparse
import sys

from elit import __version__
from elit.server.format import Input
from elit.server import parser_config


def add_model_option(parser: argparse.ArgumentParser):
    parser.add_argument('--parser_model', type=str, default=parser_config.MTL_MODEL,
                        help='the identifier of a parser model')


def main():
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    arg_parser = argparse.ArgumentParser(description='ELIT-{}'.format(__version__))
    task_parser = arg_parser.add_subparsers(dest="task", help='which task to perform?')
    parse_parser = task_parser.add_parser(name='parse', help='interactive parse per document')
    add_model_option(parse_parser)
    server_parser = task_parser.add_parser(name='serve', help='start http server',
                                           description='A http server for ELIT')
    add_model_option(server_parser)
    server_parser.add_argument('--port', type=int, default=8000)
    server_parser.add_argument('--workers', type=int, default=1, help='number of workers')

    args = arg_parser.parse_args()
    parser_config.MTL_MODEL = args.parser_model

    if args.task == 'parse':
        from elit.server.en_parser import service_parser
        for line in sys.stdin:
            line = line.strip()
            doc = service_parser.parse([Input(text=line)])[0]
            print(doc)
    elif args.task == 'serve':
        from elit.server import server
        server.run(port=args.port, workers=args.workers)


if __name__ == '__main__':
    main()
