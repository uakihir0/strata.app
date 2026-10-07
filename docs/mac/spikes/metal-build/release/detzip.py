"""The deterministic zip of docs/mac/08-build-and-distribution.md, section 11.3, as a standalone function.

    python3 detzip.py OUT.zip SOURCE_DATE_EPOCH NAME=PATH:MODE ...

Members are written in the order given, with the same date, a Unix "made by", the mode in the high 16 bits of
external_attr, and deflate level 9.  Nothing from the file system except the bytes reaches the zip.
"""
import stat
import sys
import time
import zipfile


def write_zip(out, sde, members):
    date = time.gmtime(max(int(sde), 315532800))[:6]   # zip dates start in 1980
    with zipfile.ZipFile(out, "w") as z:
        for name, path, mode in members:
            info = zipfile.ZipInfo(name, date_time=date)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)


if __name__ == "__main__":
    ms = []
    for arg in sys.argv[3:]:
        name, rest = arg.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        ms.append((name, path, int(mode, 8)))
    write_zip(sys.argv[1], sys.argv[2], ms)
