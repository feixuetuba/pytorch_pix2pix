import logging

import requests
import os
import tarfile
from argparse import ArgumentParser

def unpack(tarball, dest_dir):
    logging.info(f"Upack {tarball}")
    try:
        t = tarfile.open(tarball)
        t.extractall(path=dest_dir)
        return True
    except Exception as e:
        logging.error(e)
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--name", default="edges2handbags", help="dataset name, "
                                                                 "[edges2handbags|cityscapes|night2day|edges2shoes|facades|maps]")
    parser.add_argument("--dest", default="datasets", help="dest for save dataset")
    opts = parser.parse_args()
    os.makedirs(opts.dest, exist_ok=True)


    dest_tarball = os.path.join(opts.dest, f"{opts.name}.tar.gz")
    dest_dir = os.path.join(opts.dest, f"{opts.name}")
    if os.path.isdir(dest_dir):
        logging.info("Dataset already exist")
        exit(0)
    if os.path.isfile(dest_tarball):
        unpack(dest_tarball, dest_dir)
    else:
        logging.info(f"Downloading dataset:{opts.name}")
        url = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{opts.name}.tar.gz"
        response = requests.get(url)
        with open(dest_tarball, "wb") as fd:
            fd.write(response.content)
        unpack(dest_tarball, opts.dest)


