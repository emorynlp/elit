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
import contextlib
import glob
import gzip
import json
import logging
import os
import pickle
import platform
import random
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
import urllib
import zipfile
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Dict, Tuple, Optional, Union, List
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch
from pkg_resources import parse_version

import elit
from elit.common.constant import ELIT_URL
from elit.utils import time_util
from elit.utils.log_util import logger, flash
from elit.utils.string_util import split_long_sentence_into
from elit.utils.time_util import now_filename, CountdownTimer
from elit.version import __version__


def save_pickle(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(item: dict, path: str, ensure_ascii=False, cls=None, default=lambda o: repr(o), indent=2):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(item, out, ensure_ascii=ensure_ascii, indent=indent, cls=cls, default=default)


def load_json(path):
    with open(path, encoding='utf-8') as src:
        return json.load(src)


def load_jsonl(path, verbose=False):
    if verbose:
        src = TimingFileIterator(path)
    else:
        src = open(path, encoding='utf-8')
    for line in src:
        yield json.loads(line)
    if not verbose:
        src.close()


def filename_is_json(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension in ['.json', '.jsonl']


def make_debug_corpus(path, delimiter=None, percentage=0.1, max_samples=100):
    files = []
    if os.path.isfile(path):
        files.append(path)
    elif os.path.isdir(path):
        files += [os.path.join(path, f) for f in os.listdir(path) if
                  os.path.isfile(os.path.join(path, f)) and '.debug' not in f and not f.startswith('.')]
    else:
        raise FileNotFoundError(path)
    for filepath in files:
        filename, file_extension = os.path.splitext(filepath)
        if not delimiter:
            if file_extension in {'.tsv', '.conll', '.conllx', '.conllu'}:
                delimiter = '\n\n'
            else:
                delimiter = '\n'
        with open(filepath, encoding='utf-8') as src, open(filename + '.debug' + file_extension, 'w',
                                                           encoding='utf-8') as out:
            samples = src.read().strip().split(delimiter)
            max_samples = min(max_samples, int(len(samples) * percentage))
            out.write(delimiter.join(samples[:max_samples]))


def path_join(path, *paths):
    return os.path.join(path, *paths)


def makedirs(path):
    os.makedirs(path, exist_ok=True)
    return path


def tempdir(name=None):
    path = tempfile.gettempdir()
    if name:
        path = makedirs(path_join(path, name))
    return path


def tempdir_human():
    return tempdir(now_filename())


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    See https://interviewbubble.com/typeerror-object-of-type-float32-is-not-json-serializable/

    Args:

    Returns:

    """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def elit_home_default():
    """:return: default data directory depending on the platform and environment variables"""
    if windows():
        return os.path.join(os.environ.get('APPDATA'), 'elit')
    else:
        return os.path.join(os.path.expanduser("~"), '.elit')


def windows():
    system = platform.system()
    return system == 'Windows'


def elit_home():
    """:return: data directory in the filesystem for storage, for example when downloading models"""
    return os.getenv('elit_HOME', elit_home_default())


def file_exist(filename) -> bool:
    return os.path.isfile(filename)


def remove_file(filename):
    if file_exist(filename):
        os.remove(filename)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parent_dir(path):
    return os.path.normpath(os.path.join(path, os.pardir))


def download(url, save_path=None, save_dir=elit_home(), prefix=ELIT_URL, append_location=True, verbose=None):
    if verbose is None:
        verbose = os.environ.get('ELIT_VERBOSE', '1')
        verbose = verbose.lower() in ('1', 'true', 'yes')
    if not save_path:
        save_path = path_from_url(url, save_dir, prefix, append_location)
    if os.path.isfile(save_path):
        if verbose:
            eprint('Using local {}, ignore {}'.format(save_path, url))
        return save_path
    else:
        makedirs(parent_dir(save_path))
        if verbose:
            eprint('Downloading {} to {}'.format(url, save_path))
        tmp_path = '{}.downloading'.format(save_path)
        remove_file(tmp_path)
        try:
            def reporthook(count, block_size, total_size):
                global start_time, progress_size
                if count == 0:
                    start_time = time.time()
                    progress_size = 0
                    return
                duration = time.time() - start_time
                duration = max(1e-8, duration)
                progress_size = int(count * block_size)
                if progress_size > total_size:
                    progress_size = total_size
                speed = int(progress_size / duration)
                ratio = progress_size / total_size
                ratio = max(1e-8, ratio)
                percent = ratio * 100
                eta = duration / ratio * (1 - ratio)
                speed = human_bytes(speed)
                progress_size = human_bytes(progress_size)
                if verbose:
                    sys.stderr.write("\r%.2f%%, %s/%s, %s/s, ETA %s      " %
                                     (percent, progress_size, human_bytes(total_size), speed,
                                      time_util.report_time_delta(eta)))
                    sys.stderr.flush()

            import socket
            socket.setdefaulttimeout(10)
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', f'elit/{__version__}')]
            urllib.request.install_opener(opener)
            urlretrieve(url, tmp_path, reporthook)
            eprint()
        except BaseException as e:
            remove_file(tmp_path)
            hints_for_download = ''
            url = url.split('#')[0]
            if not windows():
                hints_for_download = f'e.g. \nwget {url} -O {save_path}\n'
            else:
                hints_for_download = ' Use some decent downloading tools.\n'
            installed_version, latest_version = check_outdated()
            if installed_version != latest_version:
                hints_for_download += f'Or upgrade to the latest version({latest_version}):\npip install -U elit'
            eprint(f'Download failed due to {repr(e)}. Please download it to {save_path} by yourself. '
                   f'{hints_for_download}')
            exit(1)
        remove_file(save_path)
        os.rename(tmp_path, save_path)
    return save_path


def parse_url_path(url):
    parsed: urllib.parse.ParseResult = urlparse(url)
    path = os.path.join(*parsed.path.strip('/').split('/'))
    return parsed.netloc, path


def uncompress(path, dest=None, remove=True, verbose=True):
    """uncompress a file

    Args:
      path: The path to a compressed file
      dest: The dest folder (Default value = None)
      remove:  (Default value = True)

    Returns:

    
    """
    # assert path.endswith('.zip')
    prefix, ext = split_if_compressed(path)
    folder_name = os.path.basename(prefix)
    file_is_zip = ext == '.zip'
    root_of_folder = None
    if ext == '.gz':
        with gzip.open(path, 'rb') as f_in, open(prefix, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        with zipfile.ZipFile(path, "r") if ext == '.zip' else tarfile.open(path, 'r:*') as archive:
            try:
                if not dest:
                    namelist = sorted(archive.namelist() if file_is_zip else archive.getnames())
                    if namelist[0] == '.':
                        namelist = namelist[1:]
                        namelist = [p[len('./'):] if p.startswith('./') else p for p in namelist]
                    if ext == '.tgz':
                        roots = set(x.split('/')[0] for x in namelist)
                        if len(roots) == 1:
                            root_of_folder = next(iter(roots))
                    else:
                        # only one file, root_of_folder = ''
                        root_of_folder = namelist[0].strip('/') if len(namelist) > 1 else ''
                    if all(f.split('/')[0] == root_of_folder for f in namelist[1:]) or not root_of_folder:
                        dest = os.path.dirname(path)  # only one folder, unzip to the same dir
                    else:
                        root_of_folder = None
                        dest = prefix  # assume zip contains more than one files or folders
                if verbose:
                    eprint('Extracting {} to {}'.format(path, dest))
                archive.extractall(dest)
                if root_of_folder:
                    if root_of_folder != folder_name:
                        # move root to match folder name
                        os.rename(path_join(dest, root_of_folder), path_join(dest, folder_name))
                    dest = path_join(dest, folder_name)
                elif len(namelist) == 1:
                    dest = path_join(dest, namelist[0])
            except (RuntimeError, KeyboardInterrupt) as e:
                remove = False
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        os.remove(dest)
                    else:
                        shutil.rmtree(dest)
                raise e
    if remove:
        remove_file(path)
    return dest


def split_if_compressed(path: str, compressed_ext=('.zip', '.tgz', '.gz', 'bz2', '.xz')) -> Tuple[str, Optional[str]]:
    tar_gz = '.tar.gz'
    if path.endswith(tar_gz):
        root, ext = path[:-len(tar_gz)], tar_gz
    else:
        root, ext = os.path.splitext(path)
    if ext in compressed_ext or ext == tar_gz:
        return root, ext
    return path, None


def get_resource(path: str, save_dir=None, extract=True, prefix=ELIT_URL, append_location=True, verbose=None):
    """Fetch real path for a resource (model, corpus, whatever)

    Args:
      path: the general path (can be a url or a real path)
      extract: whether to unzip it if it's a zip file (Default value = True)
      save_dir: return: the real path to the resource (Default value = None)
      path: str: 
      prefix:  (Default value = elit_URL)
      append_location:  (Default value = True)
      verbose: Whether print log messages.

    Returns:
      the real path to the resource

    """
    path = elit.pretrained.ALL.get(path, path)
    anchor: str = None
    compressed = None
    if verbose is None:
        verbose = os.environ.get('ELIT_VERBOSE', '1')
        verbose = verbose.lower() in ('1', 'true', 'yes')
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        pass
    elif path.startswith('http:') or path.startswith('https:'):
        url = path
        if '#' in url:
            url, anchor = url.split('#', maxsplit=1)
        realpath = path_from_url(path, save_dir, prefix, append_location)
        realpath, compressed = split_if_compressed(realpath)
        # check if resource is there
        if anchor:
            if anchor.startswith('/'):
                # indicates the folder name has to be polished
                anchor = anchor.lstrip('/')
                parts = anchor.split('/')
                realpath = str(Path(realpath).parent.joinpath(parts[0]))
                anchor = '/'.join(parts[1:])
            child = path_join(realpath, anchor)
            if os.path.exists(child):
                return child
        elif os.path.isdir(realpath) or (os.path.isfile(realpath) and (compressed and extract)):
            return realpath
        else:
            if compressed:
                pattern = realpath + '.*'
                files = glob.glob(pattern)
                files = list(filter(lambda x: not x.endswith('.downloading'), files))
                zip_path = realpath + compressed
                if zip_path in files:
                    files.remove(zip_path)
                if files:
                    if len(files) > 1:
                        logger.debug(f'Found multiple files with {pattern}, will use the first one.')
                    return files[0]
        # realpath is where its path after exaction
        if compressed:
            realpath += compressed
        if not os.path.isfile(realpath):
            path = download(url=path, save_path=realpath, verbose=verbose)
        else:
            path = realpath
    if extract and compressed:
        path = uncompress(path, verbose=verbose)
        if anchor:
            path = path_join(path, anchor)

    return path


def path_from_url(url, save_dir=elit_home(), prefix=ELIT_URL, append_location=True):
    if not save_dir:
        save_dir = elit_home()
    domain, relative_path = parse_url_path(url)
    if append_location:
        if not url.startswith(prefix):
            save_dir = os.path.join(save_dir, 'thirdparty', domain)
        else:
            # remove the relative path in prefix
            middle = prefix.split(domain)[-1].lstrip('/')
            if relative_path.startswith(middle):
                relative_path = relative_path[len(middle):]
        realpath = os.path.join(save_dir, relative_path)
    else:
        realpath = os.path.join(save_dir, os.path.basename(relative_path))
    return realpath


def human_bytes(file_size: int) -> str:
    file_size /= 1024  # KB
    if file_size > 1024:
        file_size /= 1024  # MB
        if file_size > 1024:
            file_size /= 1024  # GB
            return '%.1f GB' % file_size
        return '%.1f MB' % file_size
    return '%d KB' % file_size


def read_cells(filepath: str, delimiter='auto', strip=True, skip_first_line=False):
    filepath = get_resource(filepath)
    if delimiter == 'auto':
        if filepath.endswith('.tsv'):
            delimiter = '\t'
        elif filepath.endswith('.csv'):
            delimiter = ','
        else:
            delimiter = None
    with open(filepath, encoding='utf-8') as src:
        if skip_first_line:
            next(src)
        for line in src:
            line = line.strip()
            if not line:
                continue
            cells = line.split(delimiter)
            if strip:
                cells = [c.strip() for c in cells]
                yield cells


def replace_ext(filepath, ext) -> str:
    """Replace the extension of filepath to ext

    Args:
      filepath: 
      ext: 

    Returns:

    
    """
    file_prefix, _ = os.path.splitext(filepath)
    return file_prefix + ext


def load_word2vec(path, delimiter=' ', cache=True) -> Tuple[Dict[str, np.ndarray], int]:
    realpath = get_resource(path)
    binpath = replace_ext(realpath, '.pkl')
    if cache:
        try:
            flash('Loading word2vec from cache [blink][yellow]...[/yellow][/blink]')
            word2vec, dim = load_pickle(binpath)
            flash('')
            return word2vec, dim
        except IOError:
            pass

    dim = None
    word2vec = dict()
    f = TimingFileIterator(realpath)
    for idx, line in enumerate(f):
        f.log('Loading word2vec from text file [blink][yellow]...[/yellow][/blink]')
        line = line.rstrip().split(delimiter)
        if len(line) > 2:
            if dim is None:
                dim = len(line)
            else:
                if len(line) != dim:
                    logger.warning('{}#{} length mismatches with {}'.format(path, idx + 1, dim))
                    continue
            word, vec = line[0], line[1:]
            word2vec[word] = np.array(vec, dtype=np.float32)
    dim -= 1
    if cache:
        flash('Caching word2vec [blink][yellow]...[/yellow][/blink]')
        save_pickle((word2vec, dim), binpath)
        flash('')
    return word2vec, dim


def load_word2vec_as_vocab_tensor(path, delimiter=' ', cache=True) -> Tuple[Dict[str, int], torch.Tensor]:
    realpath = get_resource(path)
    vocab_path = replace_ext(realpath, '.vocab')
    matrix_path = replace_ext(realpath, '.pt')
    if cache:
        try:
            flash('Loading vocab and matrix from cache [blink][yellow]...[/yellow][/blink]')
            vocab = load_pickle(vocab_path)
            matrix = torch.load(matrix_path, map_location='cpu')
            flash('')
            return vocab, matrix
        except IOError:
            pass

    word2vec, dim = load_word2vec(path, delimiter, cache)
    vocab = dict((k, i) for i, k in enumerate(word2vec.keys()))
    matrix = torch.Tensor(list(word2vec.values()))
    if cache:
        flash('Caching vocab and matrix [blink][yellow]...[/yellow][/blink]')
        save_pickle(vocab, vocab_path)
        torch.save(matrix, matrix_path)
        flash('')
    return vocab, matrix


def save_word2vec(word2vec: dict, filepath, delimiter=' '):
    with open(filepath, 'w', encoding='utf-8') as out:
        for w, v in word2vec.items():
            out.write(f'{w}{delimiter}')
            out.write(f'{delimiter.join(str(x) for x in v)}\n')


def read_tsv_as_sents(tsv_file_path, ignore_prefix=None, delimiter=None):
    sent = []
    tsv_file_path = get_resource(tsv_file_path)
    with open(tsv_file_path, encoding='utf-8') as tsv_file:
        for line in tsv_file:
            if ignore_prefix and line.startswith(ignore_prefix):
                continue
            line = line.strip()
            cells = line.split(delimiter)
            if line and cells:
                sent.append(cells)
            elif sent:
                yield sent
                sent = []
    if sent:
        yield sent


def generate_words_tags_from_tsv(tsv_file_path, lower=False, gold=True, max_seq_length=None, sent_delimiter=None,
                                 char_level=False, hard_constraint=False):
    for sent in read_tsv_as_sents(tsv_file_path):
        words = [cells[0] for cells in sent]
        if max_seq_length:
            offset = 0
            # try to split the sequence to make it fit into max_seq_length
            for shorter_words in split_long_sentence_into(words, max_seq_length, sent_delimiter, char_level,
                                                          hard_constraint):
                if gold:
                    shorter_tags = [cells[1] for cells in sent[offset:offset + len(shorter_words)]]
                    offset += len(shorter_words)
                else:
                    shorter_tags = None
                if lower:
                    shorter_words = [word.lower() for word in shorter_words]
                yield shorter_words, shorter_tags
        else:
            if gold:
                try:
                    tags = [cells[1] for cells in sent]
                except:
                    raise ValueError(f'Failed to load {tsv_file_path}: {sent}')
            else:
                tags = None
            if lower:
                words = [word.lower() for word in words]
            yield words, tags


def prune_ner_tagset(sample: dict, tagset: Union[set, Dict[str, str]]):
    if 'tag' in sample:
        pruned_tag = []
        for tag in sample['tag']:
            cells = tag.split('-', 1)
            if len(cells) == 2:
                role, ner_type = cells
                if ner_type in tagset:
                    if isinstance(tagset, dict):
                        tag = role + '-' + tagset[ner_type]
                else:
                    tag = 'O'
            pruned_tag.append(tag)
        sample['tag'] = pruned_tag
    return sample


def split_file(filepath, train=0.8, dev=0.1, test=0.1, names=None, shuffle=False):
    num_samples = 0
    if filepath.endswith('.tsv'):
        for sent in read_tsv_as_sents(filepath):
            num_samples += 1
    else:
        with open(filepath, encoding='utf-8') as src:
            for sample in src:
                num_samples += 1
    splits = {'train': train, 'dev': dev, 'test': test}
    splits = dict((k, v) for k, v in splits.items() if v)
    splits = dict((k, v / sum(splits.values())) for k, v in splits.items())
    accumulated = 0
    r = []
    for k, v in splits.items():
        r.append(accumulated)
        accumulated += v
        r.append(accumulated)
        splits[k] = accumulated
    if names is None:
        names = {}
    name, ext = os.path.splitext(filepath)
    filenames = [names.get(split, name + '.' + split + ext) for split in splits.keys()]
    outs = [open(f, 'w', encoding='utf-8') for f in filenames]
    if shuffle:
        shuffle = list(range(num_samples))
        random.shuffle(shuffle)
    if filepath.endswith('.tsv'):
        src = read_tsv_as_sents(filepath)
    else:
        src = open(filepath, encoding='utf-8')
    for idx, sample in enumerate(src):
        if shuffle:
            idx = shuffle[idx]
        ratio = idx / num_samples
        for sid, out in enumerate(outs):
            if r[2 * sid] <= ratio < r[2 * sid + 1]:
                if isinstance(sample, list):
                    sample = '\n'.join('\t'.join(x) for x in sample) + '\n\n'
                out.write(sample)
                break
    if not filepath.endswith('.tsv'):
        src.close()
    for out in outs:
        out.close()
    return filenames


def fileno(file_or_fd):
    try:
        fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    except:
        return None
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """Redirect stdout to else where
    Copied from https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262

    Args:
      to:  (Default value = os.devnull)
      stdout:  (Default value = None)

    Returns:

    
    """
    if windows():  # This doesn't play well with windows
        yield None
        return
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)
    if not stdout_fd:
        yield None
        return
        # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            try:
                stdout.flush()
                os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            except:
                # This is the best we can do
                pass


def get_exitcode_stdout_stderr(cmd):
    """Execute the external command and get its exitcode, stdout and stderr.
    See https://stackoverflow.com/a/21000308/3730690

    Args:
      cmd: 

    Returns:

    
    """
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out.decode('utf-8'), err.decode('utf-8')


def run_cmd(cmd: str) -> str:
    exitcode, out, err = get_exitcode_stdout_stderr(cmd)
    if exitcode:
        raise RuntimeError(err + '\nThe command is:\n' + cmd)
    return out


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def basename_no_ext(path):
    basename = os.path.basename(path)
    no_ext, ext = os.path.splitext(basename)
    return no_ext


def file_cache(path: str, purge=False):
    cache_name = path + '.cache'
    cache_time = os.path.getmtime(cache_name) if os.path.isfile(cache_name) and not purge else 0
    file_time = os.path.getmtime(path)
    cache_valid = cache_time > file_time
    return cache_name, cache_valid


def merge_files(files: List[str], dst: str):
    with open(dst, 'wb') as write:
        for f in files:
            with open(f, 'rb') as read:
                shutil.copyfileobj(read, write)


class TimingFileIterator(CountdownTimer):

    def __init__(self, filepath) -> None:
        super().__init__(os.path.getsize(filepath))
        self.filepath = filepath

    def __iter__(self):
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(self.filepath)
        fp = open(self.filepath, encoding='utf-8', errors='ignore')
        line = fp.readline()
        while line:
            yield line
            self.current = fp.tell()
            line = fp.readline()
        fp.close()

    def log(self, info=None, ratio_percentage=True, ratio=True, step=0, interval=0.5, erase=True,
            logger: Union[logging.Logger, bool] = None, newline=False, ratio_width=None):
        assert step == 0
        super().log(info, ratio_percentage, ratio, step, interval, erase, logger, newline, ratio_width)

    @property
    def ratio(self) -> str:
        return f'{human_bytes(self.current)}/{human_bytes(self.total)}'

    @property
    def ratio_width(self) -> int:
        return len(f'{human_bytes(self.total)}') * 2 + 1

    def close(self):
        pass


def check_outdated(package='elit', version=__version__, repository_url='https://pypi.python.org/pypi/%s/json'):
    """
    Given the name of a package on PyPI and a version (both strings), checks
    if the given version is the latest version of the package available.
    Returns a 2-tuple (installed_version, latest_version)
    `repository_url` is a `%` style format string
    to use a different repository PyPI repository URL,
    e.g. test.pypi.org or a private repository.
    The string is formatted with the package name.
    Adopted from https://github.com/alexmojaki/outdated/blob/master/outdated/__init__.py
    """

    installed_version = parse_version(version)
    latest_version = get_latest_info_from_pypi(package, repository_url)
    return installed_version, latest_version


def get_latest_info_from_pypi(package='elit', repository_url='https://pypi.python.org/pypi/%s/json'):
    url = repository_url % package
    response = urllib.request.urlopen(url).read()
    return parse_version(json.loads(response)['info']['version'])
