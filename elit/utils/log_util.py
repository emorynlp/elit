# ========================================================================
# Copyright 2020 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

# -*- coding:utf-8 -*-
# Author: hankcs
import datetime
import io
import logging
import os
import sys
from logging import LogRecord

import termcolor


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', enable=True):
        super().__init__(fmt, datefmt, style)
        self.enable = enable

    def formatMessage(self, record: LogRecord) -> str:
        message = super().formatMessage(record)
        if self.enable:
            return color_format(message)
        else:
            return remove_color_tag(message)


def init_logger(name=None, root_dir=None, level=logging.INFO, mode='w',
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S') -> logging.Logger:
    if not name:
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    rootLogger = logging.getLogger(os.path.join(root_dir, name) if root_dir else name)
    rootLogger.propagate = False

    consoleHandler = logging.StreamHandler(sys.stdout)  # stderr will be rendered as red which is bad
    consoleHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    attached_to_std = False
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(level)
    consoleHandler.setLevel(level)

    if root_dir:
        os.makedirs(root_dir, exist_ok=True)
        log_path = "{0}/{1}.log".format(root_dir, name)
        fileHandler = logging.FileHandler(log_path, mode=mode)
        fileHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt, enable=False))
        rootLogger.addHandler(fileHandler)
        fileHandler.setLevel(level)

    return rootLogger


logger = init_logger(name='elit', level=os.environ.get('elit_LOG_LEVEL', 'INFO'))


def enable_debug(debug=True):
    logger.setLevel(logging.DEBUG if debug else logging.ERROR)


class ErasablePrinter(object):
    def __init__(self):
        self._last_print_width = 0

    def erase(self):
        if self._last_print_width:
            sys.stdout.write("\b" * self._last_print_width)
            sys.stdout.write(" " * self._last_print_width)
            sys.stdout.write("\b" * self._last_print_width)
            sys.stdout.write("\r")  # \r is essential when multi-lines were printed
            self._last_print_width = 0

    def print(self, msg: str, color=True):
        self.erase()
        if color:
            msg, _len = color_format_len(msg)
            self._last_print_width = _len
        else:
            self._last_print_width = len(msg)
        sys.stdout.write(msg)
        sys.stdout.flush()


_printer = ErasablePrinter()


def flash(line: str, color=True):
    _printer.print(line, color)


def color_format(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg = msg.replace(start, '\033[%dm' % v).replace(end, termcolor.RESET)
    return msg


def remove_color_tag(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg = msg.replace(start, '').replace(end, '')
    return msg


def color_format_len(msg: str):
    _len = len(msg)
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg, delta = _replace_color_offset(msg, start, '\033[%dm' % v)
            _len -= delta
            msg, delta = _replace_color_offset(msg, end, termcolor.RESET)
            _len -= delta
    return msg, _len


def _replace_color_offset(msg: str, color: str, ctrl: str):
    chunks = msg.split(color)
    delta = (len(chunks) - 1) * len(color)
    return ctrl.join(chunks), delta


def cprint(*args, **kwargs):
    out = io.StringIO()
    print(*args, file=out, **kwargs)
    text = out.getvalue()
    out.close()
    c_text = color_format(text)
    print(c_text, end='')


def main():
    # cprint('[blink][yellow]...[/yellow][/blink]')
    # show_colors_and_formats()
    show_colors()
    # print('previous', end='')
    # for i in range(10):
    #     flash(f'[red]{i}[/red]')


def show_colors_and_formats():
    msg = ''
    for c in termcolor.COLORS.keys():
        for h in termcolor.HIGHLIGHTS.keys():
            for a in termcolor.ATTRIBUTES.keys():
                msg += f'[{c}][{h}][{a}] {c}+{h}+{a} [/{a}][/{h}][/{c}]'
    logger.info(msg)


def show_colors():
    msg = ''
    for c in termcolor.COLORS.keys():
        cprint(f'[{c}]"{c}",[/{c}]')


# Generates tables for Doxygen flavored Markdown.  See the Doxygen
# documentation for details:
#   http://www.doxygen.nl/manual/markdown.html#md_tables

# Translation dictionaries for table alignment
left_rule = {'<': ':', '^': ':', '>': '-'}
right_rule = {'<': '-', '^': ':', '>': ':'}


def evalute_field(record, field_spec):
    """Evalute a field of a record using the type of the field_spec as a guide.

    Args:
      record: 
      field_spec: 

    Returns:

    """
    if type(field_spec) is int:
        return str(record[field_spec])
    elif type(field_spec) is str:
        return str(getattr(record, field_spec))
    else:
        return str(field_spec(record))


def markdown_table(headings, records, fields=None, alignment=None, file=None):
    """Generate a Doxygen-flavor Markdown table from records.
    See https://stackoverflow.com/questions/13394140/generate-markdown-tables
    
    file -- Any object with a 'write' method that takes a single string
        parameter.
    records -- Iterable.  Rows will be generated from this.
    fields -- List of fields for each row.  Each entry may be an integer,
        string or a function.  If the entry is an integer, it is assumed to be
        an index of each record.  If the entry is a string, it is assumed to be
        a field of each record.  If the entry is a function, it is called with
        the record and its return value is taken as the value of the field.
    headings -- List of column headings.
    alignment - List of pairs alignment characters.  The first of the pair
        specifies the alignment of the header, (Doxygen won't respect this, but
        it might look good, the second specifies the alignment of the cells in
        the column.
    
        Possible alignment characters are:
            '<' = Left align
            '>' = Right align (default for cells)
            '^' = Center (default for column headings)

    Args:
      headings: 
      records: 
      fields:  (Default value = None)
      alignment:  (Default value = None)
      file:  (Default value = None)

    Returns:

    """
    if not file:
        file = io.StringIO()
    num_columns = len(headings)
    if not fields:
        fields = list(range(num_columns))
    assert len(headings) == num_columns

    # Compute the table cell data
    columns = [[] for i in range(num_columns)]
    for record in records:
        for i, field in enumerate(fields):
            columns[i].append(evalute_field(record, field))

    # Fill out any missing alignment characters.
    extended_align = alignment if alignment is not None else [('^', '<')]
    if len(extended_align) > num_columns:
        extended_align = extended_align[0:num_columns]
    elif len(extended_align) < num_columns:
        extended_align += [('^', '>') for i in range(num_columns - len(extended_align))]

    heading_align, cell_align = [x for x in zip(*extended_align)]

    field_widths = [len(max(column, key=len)) if len(column) > 0 else 0
                    for column in columns]
    heading_widths = [max(len(head), 2) for head in headings]
    column_widths = [max(x) for x in zip(field_widths, heading_widths)]

    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(heading_align, column_widths)])
    heading_template = '| ' + _ + ' |'
    _ = ' | '.join(['{:' + a + str(w) + '}'
                    for a, w in zip(cell_align, column_widths)])
    row_template = '| ' + _ + ' |'

    _ = ' | '.join([left_rule[a] + '-' * (w - 2) + right_rule[a]
                    for a, w in zip(cell_align, column_widths)])
    ruling = '| ' + _ + ' |'

    file.write(heading_template.format(*headings).rstrip() + '\n')
    file.write(ruling.rstrip() + '\n')
    for row in zip(*columns):
        file.write(row_template.format(*row).rstrip() + '\n')
    if isinstance(file, io.StringIO):
        text = file.getvalue()
        file.close()
        return text


if __name__ == '__main__':
    main()
