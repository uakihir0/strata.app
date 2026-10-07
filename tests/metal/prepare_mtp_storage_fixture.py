#!/usr/bin/env python3
"""Safely prepare the dedicated synthetic MTP storage fixture under the build tree."""

from __future__ import annotations

import ctypes
import errno
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile

_FIXTURE_PREFIX = "mtp-storage-fixture-v2-"
_FIXTURE_NAME = _FIXTURE_PREFIX + "selftest"
_OWNER_NAME = ".strata-owned-fixture"
_OWNER_CONTENT = b"strata MTP storage CTest fixture v1\n"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_ACL_TYPE_EXTENDED = 0x00000100
_ACL_LIBRARY = None


class FixtureError(RuntimeError):
    """A safe fixture operation could not proceed."""


def _verify_no_extended_acl(directory_fd: int, description: str) -> None:
    global _ACL_LIBRARY
    try:
        if _ACL_LIBRARY is None:
            _ACL_LIBRARY = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            _ACL_LIBRARY.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
            _ACL_LIBRARY.acl_get_fd_np.restype = ctypes.c_void_p
            _ACL_LIBRARY.acl_free.argtypes = [ctypes.c_void_p]
            _ACL_LIBRARY.acl_free.restype = ctypes.c_int
    except (AttributeError, OSError) as error:
        raise FixtureError(f"cannot verify the macOS ACL for {description}: {error}") from error

    ctypes.set_errno(0)
    acl = _ACL_LIBRARY.acl_get_fd_np(directory_fd, _ACL_TYPE_EXTENDED)
    acl_error = ctypes.get_errno()
    if acl:
        free_result = _ACL_LIBRARY.acl_free(acl)
        if free_result != 0:
            raise FixtureError(f"cannot release the inspected ACL for {description}")
        raise FixtureError(f"{description} has an extended ACL")
    if acl_error != errno.ENOENT:
        message = os.strerror(acl_error) if acl_error else "ACL state is indeterminate"
        raise FixtureError(f"cannot verify the macOS ACL for {description}: {message}")


def _absolute_directory_fd(path: str) -> int:
    """Open every path component without following symlinks."""
    normalized = os.path.normpath(os.path.abspath(path))
    if not normalized.startswith(os.path.sep):
        raise FixtureError("build root must be an absolute path")
    descriptor = os.open(os.path.sep, _DIRECTORY_FLAGS)
    try:
        for component in normalized.split(os.path.sep):
            if not component:
                continue
            next_descriptor = os.open(
                component, _DIRECTORY_FLAGS, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_fixture_parent(directory_fd: int) -> None:
    status = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise FixtureError(
            "fixture parent must be a directory owned by the current user "
            "without group or world write access"
        )
    _verify_no_extended_acl(directory_fd, "fixture parent")


def _verify_fixture_directory(directory_fd: int) -> os.stat_result:
    status = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or status.st_mode & 0o077
    ):
        raise FixtureError(
            "fixture directory must be owned by the current user with mode 0700"
        )
    _verify_no_extended_acl(directory_fd, "fixture directory")
    return status


def _open_build_parent(build_root: str) -> tuple[int, str]:
    """Open build/metal as a directory handle, creating it without path traversal."""
    resolved_root = os.path.realpath(os.path.abspath(build_root))
    root_fd = _absolute_directory_fd(resolved_root)
    parent_fd = -1
    try:
        try:
            parent_fd = os.open("metal", _DIRECTORY_FLAGS, dir_fd=root_fd)
        except FileNotFoundError:
            try:
                os.mkdir("metal", 0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            parent_fd = os.open("metal", _DIRECTORY_FLAGS, dir_fd=root_fd)
        _verify_fixture_parent(parent_fd)
        return parent_fd, resolved_root
    except (OSError, FixtureError) as error:
        if parent_fd >= 0:
            os.close(parent_fd)
        raise FixtureError(
            f"refusing an unsafe MTP fixture parent directory: {error}"
        ) from error
    finally:
        os.close(root_fd)


def _validate_fixture_path(build_root: str, fixture_dir: str) -> tuple[str, str]:
    root_input = os.path.normpath(os.path.abspath(build_root))
    requested = os.path.normpath(os.path.abspath(fixture_dir))
    fixture_name = os.path.basename(requested)
    expected = os.path.join(root_input, "metal", fixture_name)
    if (
        requested != expected
        or re.fullmatch(re.escape(_FIXTURE_PREFIX) + r"[0-9a-f]{32}", fixture_name) is None
    ):
        raise FixtureError("fixture path must be a unique dedicated build/metal directory")
    return os.path.realpath(root_input), fixture_name


def _verify_owner(directory_fd: int) -> None:
    try:
        owner_fd = os.open(_OWNER_NAME, _FILE_FLAGS, dir_fd=directory_fd)
    except OSError as error:
        raise FixtureError(
            "MTP fixture has no valid ownership marker"
        ) from error
    try:
        status = os.fstat(owner_fd)
        if not stat.S_ISREG(status.st_mode):
            raise FixtureError("MTP fixture ownership marker is not a regular file")
        with os.fdopen(os.dup(owner_fd), "rb") as marker:
            if marker.read(len(_OWNER_CONTENT) + 1) != _OWNER_CONTENT:
                raise FixtureError("MTP fixture ownership marker is invalid")
    finally:
        os.close(owner_fd)


def _verify_empty_owned_fixture(directory_fd: int) -> None:
    if os.listdir(directory_fd):
        raise FixtureError("new MTP fixture directory already contains data")


def _verify_only_owner_marker(directory_fd: int) -> None:
    if sorted(os.listdir(directory_fd)) != [_OWNER_NAME]:
        raise FixtureError("MTP fixture changed during ownership initialization")


def _write_owner_marker(directory_fd: int) -> os.stat_result:
    owner_fd = os.open(
        _OWNER_NAME,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        owner_status = os.fstat(owner_fd)
        offset = 0
        while offset < len(_OWNER_CONTENT):
            count = os.write(owner_fd, _OWNER_CONTENT[offset:])
            if count <= 0:
                raise OSError("short write while creating MTP fixture ownership marker")
            offset += count
        os.fsync(owner_fd)
    finally:
        os.close(owner_fd)
    return owner_status


def _create_owned_fixture_directory(
    parent_fd: int, name: str
) -> tuple[int, os.stat_result]:
    _verify_fixture_parent(parent_fd)
    # mkdirat does not return a descriptor. A private, current-user-owned
    # parent prevents other users from replacing the entry before it is pinned;
    # a same-UID process is within this build directory's trust boundary.
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        # Without a pinned descriptor the entry may have been replaced, so do
        # not attempt path-based cleanup.
        raise FixtureError(f"cannot pin the new MTP fixture directory: {error}") from error
    try:
        opened_status = _verify_fixture_directory(directory_fd)
        try:
            current_status = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FixtureError("new MTP fixture path disappeared while pinning it") from error
        if not stat.S_ISDIR(current_status.st_mode) or (
            opened_status.st_dev,
            opened_status.st_ino,
        ) != (current_status.st_dev, current_status.st_ino):
            raise FixtureError("new MTP fixture directory changed while opening")
        _verify_empty_owned_fixture(directory_fd)
        _write_owner_marker(directory_fd)
        _verify_owner(directory_fd)
        _verify_only_owner_marker(directory_fd)
        return directory_fd, opened_status
    except BaseException:
        os.close(directory_fd)
        raise


def _verify_created_fixture(directory_fd: int) -> None:
    _verify_fixture_directory(directory_fd)
    _verify_owner(directory_fd)
    if len(os.listdir(directory_fd)) <= 1:
        raise FixtureError("MTP fixture generator did not write the runtime files")


def _with_prepared_fixture(build_root: str, fixture_tool: str, action):
    resolved_root = os.path.realpath(os.path.abspath(build_root))
    parent_fd, opened_root = _open_build_parent(resolved_root)
    try:
        if opened_root != resolved_root:
            raise FixtureError("build root changed while opening the fixture parent")
        fixture_name = _FIXTURE_PREFIX + secrets.token_hex(16)
        fixture_dir = os.path.join(resolved_root, "metal", fixture_name)
        _validate_fixture_path(resolved_root, fixture_dir)
        directory_fd, directory_status = _create_owned_fixture_directory(
            parent_fd, fixture_name
        )
        try:
            result = subprocess.run(
                [
                    fixture_tool,
                    "--create-fixture-with-mtp-in-dir-fd",
                    str(directory_fd),
                    str(directory_status.st_dev),
                    str(directory_status.st_ino),
                ],
                pass_fds=(directory_fd,),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise FixtureError(
                    f"fixture generator exited with status {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
            _verify_created_fixture(directory_fd)
            return action(directory_fd, directory_status, fixture_name)
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def _prepare(build_root: str, fixture_tool: str) -> str:
    return _with_prepared_fixture(
        build_root, fixture_tool, lambda _fd, _status, name: name
    )


def _run_storage_test(
    build_root: str,
    fixture_tool: str,
    storage_test: str,
    metallib: str,
    engine: str,
) -> str:
    def run_pinned_test(
        directory_fd: int, directory_status: os.stat_result, fixture_name: str
    ) -> str:
        result = subprocess.run(
            [
                storage_test,
                metallib,
                "--fixture-dir-fd",
                str(directory_fd),
                str(directory_status.st_dev),
                str(directory_status.st_ino),
            ],
            pass_fds=(directory_fd,),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FixtureError(
                f"MTP storage test failed for pinned fixture {fixture_name} "
                f"with status {result.returncode}: {result.stderr.strip()}"
            )
        storage_args = [
            "--storage-check",
            "--native",
            os.path.join("metal", fixture_name, "tiny.gguf"),
            "--pack",
            os.path.join("metal", fixture_name, "pack"),
            "--mtp",
            os.path.join("metal", fixture_name, "mtp-rt"),
            "--max-context",
            "13",
            "--kv",
            "int8",
            "--vram-reserve-mib",
            "0",
            "--mtp-window",
            "8",
            "--prefill",
            "9",
        ]
        full_speculative = subprocess.run(
            [engine, *storage_args, "--spec", "8"],
            cwd=build_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            full_speculative.returncode != 0
            or "STORAGE_CHECK ok mode=full" not in full_speculative.stdout
            or "main_kv=yes" not in full_speculative.stdout
            or "ple_table=yes" not in full_speculative.stdout
            or "window_scratch=yes" not in full_speculative.stdout
            or "mtp_runtime=yes" not in full_speculative.stdout
        ):
            raise FixtureError(
                f"full speculative --storage-check failed for pinned fixture {fixture_name} "
                f"with status {full_speculative.returncode}: "
                f"{full_speculative.stderr.strip()}"
            )
        if "inference=unavailable" not in full_speculative.stdout:
            raise FixtureError("speculative --storage-check did not preserve storage-only status")

        tiered_speculative = subprocess.run(
            [engine, *storage_args, "--expert-cache", "0", "--spec", "8"],
            cwd=build_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            tiered_speculative.returncode != 0
            or "STORAGE_CHECK ok mode=tiered" not in tiered_speculative.stdout
            or "main_kv=yes" not in tiered_speculative.stdout
            or "ple_table=no" not in tiered_speculative.stdout
            or "window_scratch=yes" not in tiered_speculative.stdout
            or "mtp_runtime=yes" not in tiered_speculative.stdout
        ):
            raise FixtureError(
                f"tiered speculative --storage-check failed for pinned fixture {fixture_name} "
                f"with status {tiered_speculative.returncode}: "
                f"{tiered_speculative.stderr.strip()}"
            )

        ordinary = subprocess.run(
            [engine, *storage_args, "--expert-cache", "0"],
            cwd=build_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if (ordinary.returncode != 0 or "main_kv=yes" not in ordinary.stdout
                or "ple_table=no" not in ordinary.stdout
                or "window_scratch=yes" not in ordinary.stdout
                or "mtp_runtime=no" not in ordinary.stdout):
            raise FixtureError(
                f"non-speculative --storage-check opened an invalid storage plan for "
                f"fixture {fixture_name} (status {ordinary.returncode}): "
                f"{ordinary.stderr.strip()}"
            )
        return (
            result.stdout
            + full_speculative.stdout
            + tiered_speculative.stdout
            + ordinary.stdout
        )

    return _with_prepared_fixture(build_root, fixture_tool, run_pinned_test)


def _open_weight_cli_fixture_parent(build_root: str, parent_dir: str) -> int:
    root_input = os.path.normpath(os.path.abspath(build_root))
    parent_input = os.path.normpath(os.path.abspath(parent_dir))
    expected_prefix = os.path.join(root_input, "metal", "weight-store-cli")
    parent_base = os.path.basename(parent_input)
    if (
        os.path.dirname(parent_input) != expected_prefix
        or re.fullmatch(r"fixtures-[0-9a-f]{32}", parent_base) is None
    ):
        raise FixtureError("CLI fixture parent is outside its dedicated build directory")

    root_fd = _absolute_directory_fd(os.path.realpath(root_input))
    descriptors = [root_fd]
    fixture_parent_fd = -1
    keep_fixture_parent = False
    try:
        metal_fd = os.open("metal", _DIRECTORY_FLAGS, dir_fd=root_fd)
        descriptors.append(metal_fd)
        cli_root_fd = os.open("weight-store-cli", _DIRECTORY_FLAGS, dir_fd=metal_fd)
        descriptors.append(cli_root_fd)
        fixture_parent_fd = os.open(parent_base, _DIRECTORY_FLAGS, dir_fd=cli_root_fd)
        descriptors.append(fixture_parent_fd)
        _verify_fixture_parent(fixture_parent_fd)
        keep_fixture_parent = True
        return fixture_parent_fd
    except OSError as error:
        raise FixtureError(f"cannot securely open CLI fixture parent: {error}") from error
    finally:
        for descriptor in reversed(descriptors):
            if not keep_fixture_parent or descriptor != fixture_parent_fd:
                os.close(descriptor)


def _create_weight_cli_child(
    build_root: str, parent_dir: str, child_name: str, fixture_tool: str
) -> None:
    if (
        not child_name
        or os.path.basename(child_name) != child_name
        or child_name in {".", ".."}
    ):
        raise FixtureError("CLI fixture child must be one directory name")
    parent_fd = _open_weight_cli_fixture_parent(build_root, parent_dir)
    try:
        try:
            child_status = os.stat(
                child_name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(child_status.st_mode):
                raise FixtureError("MTP fixture destination must not be a symlink")
            if stat.S_ISDIR(child_status.st_mode):
                try:
                    existing_fd = os.open(
                        child_name, _DIRECTORY_FLAGS, dir_fd=parent_fd
                    )
                except OSError as error:
                    raise FixtureError(
                        f"cannot safely inspect existing CLI fixture child: {error}"
                    ) from error
                try:
                    if os.listdir(existing_fd):
                        raise FixtureError(
                            "MTP fixture destination must be an empty directory"
                        )
                finally:
                    os.close(existing_fd)
            raise FixtureError("CLI fixture child path already exists")
        directory_fd, directory_status = _create_owned_fixture_directory(
            parent_fd, child_name
        )
        try:
            result = subprocess.run(
                [
                    fixture_tool,
                    "--create-fixture-with-mtp-in-dir-fd",
                    str(directory_fd),
                    str(directory_status.st_dev),
                    str(directory_status.st_ino),
                ],
                pass_fds=(directory_fd,),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise FixtureError(
                    f"fixture generator exited with status {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
            _verify_created_fixture(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def _expect_refusal(action, description: str) -> None:
    try:
        action()
    except (FixtureError, OSError):
        return
    raise FixtureError(f"fixture safety self-test did not refuse {description}")


def _add_fixture_acl_for_self_test(path: str) -> None:
    result = subprocess.run(
        [
            "/bin/chmod",
            "+a",
            "everyone allow add_file,add_subdirectory,delete_child",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FixtureError(
            "could not create the ACL safety self-test: "
            f"{result.stderr.strip()}"
        )


def _run_self_tests(fixture_tool: str) -> None:
    with tempfile.TemporaryDirectory(prefix="strata-mtp-fixture-check-") as temporary:
        root = os.path.join(temporary, "build")
        parent = os.path.join(root, "metal")
        external = os.path.join(temporary, "external")
        os.makedirs(parent)
        os.mkdir(external)
        sentinel = os.path.join(external, "sentinel")
        with open(sentinel, "wb") as output:
            output.write(b"keep\n")

        broad_sentinel = os.path.join(parent, "sentinel")
        with open(broad_sentinel, "wb") as output:
            output.write(b"broad path\n")
        _expect_refusal(
            lambda: _validate_fixture_path(root, parent),
            "a broad build/metal cleanup path",
        )
        with open(broad_sentinel, "rb") as source:
            if source.read() != b"broad path\n":
                raise FixtureError("broad-path refusal changed its sentinel")

        parent_link_root = os.path.join(temporary, "parent-link-build")
        os.mkdir(parent_link_root)
        os.symlink(external, os.path.join(parent_link_root, "metal"))
        _expect_refusal(
            lambda: _open_build_parent(parent_link_root),
            "a symlinked fixture parent",
        )
        with open(sentinel, "rb") as source:
            if source.read() != b"keep\n":
                raise FixtureError("parent-symlink refusal changed its external sentinel")

        insecure_parent_root = os.path.join(temporary, "insecure-parent-build")
        insecure_parent = os.path.join(insecure_parent_root, "metal")
        os.makedirs(insecure_parent)
        os.chmod(insecure_parent, 0o777)
        insecure_parent_fd = _absolute_directory_fd(os.path.realpath(insecure_parent))
        try:
            _expect_refusal(
                lambda: _create_owned_fixture_directory(
                    insecure_parent_fd, _FIXTURE_NAME
                ),
                "a group/world-writable fixture parent",
            )
            if os.listdir(insecure_parent_fd):
                raise FixtureError("insecure fixture parent refusal created an entry")
        finally:
            os.close(insecure_parent_fd)

        acl_parent_root = os.path.join(temporary, "acl-parent-build")
        acl_parent = os.path.join(acl_parent_root, "metal")
        os.makedirs(acl_parent, mode=0o700)
        _add_fixture_acl_for_self_test(acl_parent)
        acl_parent_fd = _absolute_directory_fd(os.path.realpath(acl_parent))
        try:
            if os.fstat(acl_parent_fd).st_mode & 0o077:
                raise FixtureError("ACL parent self-test unexpectedly changed mode bits")
            _expect_refusal(
                lambda: _create_owned_fixture_directory(
                    acl_parent_fd, _FIXTURE_NAME
                ),
                "an ACL-enabled fixture parent",
            )
            if os.listdir(acl_parent_fd):
                raise FixtureError("ACL parent refusal changed its contents")
        finally:
            os.close(acl_parent_fd)

        acl_race_root = os.path.join(temporary, "acl-race-build")
        os.makedirs(os.path.join(acl_race_root, "metal"))
        acl_race_parent_fd, _ = _open_build_parent(acl_race_root)
        acl_race_name = "acl-added-after-pin"
        acl_race_path = os.path.join(acl_race_root, "metal", acl_race_name)
        acl_marker_path = os.path.join(acl_race_path, _OWNER_NAME)
        original_open = os.open
        acl_race_fd: int | None = None
        acl_added_after_pin = False

        def add_acl_after_directory_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal acl_race_fd, acl_added_after_pin
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if (
                not acl_added_after_pin
                and path == acl_race_name
                and dir_fd == acl_race_parent_fd
                and flags & getattr(os, "O_DIRECTORY", 0)
            ):
                acl_race_fd = descriptor
                _add_fixture_acl_for_self_test(acl_race_path)
                if os.fstat(descriptor).st_mode & 0o077:
                    raise FixtureError(
                        "post-pin ACL self-test unexpectedly changed mode bits"
                    )
                acl_added_after_pin = True
            return descriptor

        try:
            os.open = add_acl_after_directory_open
            _expect_refusal(
                lambda: _create_owned_fixture_directory(
                    acl_race_parent_fd, acl_race_name
                ),
                "an ACL added after the fixture directory was pinned",
            )
        finally:
            os.open = original_open
            os.close(acl_race_parent_fd)
        if not acl_added_after_pin or acl_race_fd is None:
            raise FixtureError("post-pin ACL self-test did not apply its ACL")
        if os.path.exists(acl_marker_path):
            raise FixtureError("fixture ownership marker was created after ACL refusal")
        try:
            os.fstat(acl_race_fd)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
        else:
            raise FixtureError("post-pin ACL refusal leaked its directory descriptor")

        final_link_root = os.path.join(temporary, "final-link-build")
        final_link_parent = os.path.join(final_link_root, "metal")
        os.makedirs(final_link_parent)
        final_link = os.path.join(final_link_parent, _FIXTURE_NAME)
        os.symlink(external, final_link)
        parent_fd, _ = _open_build_parent(final_link_root)
        try:
            generator_result = subprocess.run(
                [
                    fixture_tool,
                    "--create-fixture-with-mtp-at-fd",
                    str(parent_fd),
                    _FIXTURE_NAME,
                ],
                pass_fds=(parent_fd,),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if generator_result.returncode == 0:
                raise FixtureError(
                    "descriptor-relative generator accepted a symlink target"
                )
        finally:
            os.close(parent_fd)
        with open(sentinel, "rb") as source:
            if source.read() != b"keep\n":
                raise FixtureError("final-symlink refusal changed its external sentinel")

        final_race_root = os.path.join(temporary, "final-race-build")
        final_race_parent = os.path.join(final_race_root, "metal")
        os.makedirs(final_race_parent)
        parent_fd, _ = _open_build_parent(final_race_root)
        try:
            pin_result = subprocess.run(
                [
                    fixture_tool,
                    "--test-mtp-fixture-pin",
                    str(parent_fd),
                    "pin-race-probe",
                    external,
                ],
                pass_fds=(parent_fd,),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if pin_result.returncode != 0:
                raise FixtureError("final-directory pinning self-test failed")
        finally:
            os.close(parent_fd)
        with open(sentinel, "rb") as source:
            if source.read() != b"keep\n":
                raise FixtureError("final-path replacement wrote through its external symlink")
        with open(
            os.path.join(final_race_parent, "pin-race-probe.held", "pinned-write"),
            "rb",
        ) as source:
            if source.read() != b"pinned\n":
                raise FixtureError("final-path replacement escaped its pinned directory")

        fifo_owner_root = os.path.join(temporary, "fifo-owner-build")
        fifo_owner_fixture = os.path.join(fifo_owner_root, "metal", _FIXTURE_NAME)
        os.makedirs(fifo_owner_fixture)
        os.mkfifo(os.path.join(fifo_owner_fixture, _OWNER_NAME), 0o600)
        fifo_owner_fd = os.open(
            fifo_owner_fixture, _DIRECTORY_FLAGS
        )
        try:
            _expect_refusal(
                lambda: _verify_owner(fifo_owner_fd),
                "a FIFO ownership marker",
            )
        finally:
            os.close(fifo_owner_fd)

        replacement_root = os.path.join(temporary, "replacement-build")
        os.makedirs(os.path.join(replacement_root, "metal"))
        parent_fd, _ = _open_build_parent(replacement_root)
        original_open = os.open
        replaced_directory = False
        replacement_payload = os.path.join(
            replacement_root, "metal", _FIXTURE_NAME, "sentinel"
        )

        def replace_after_directory_pin(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal replaced_directory
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if (
                not replaced_directory
                and path == _FIXTURE_NAME
                and dir_fd == parent_fd
                and flags & getattr(os, "O_DIRECTORY", 0)
            ):
                os.rename(
                    _FIXTURE_NAME,
                    _FIXTURE_NAME + ".held",
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.mkdir(_FIXTURE_NAME, 0o700, dir_fd=parent_fd)
                replacement_fd = original_open(
                    _FIXTURE_NAME, _DIRECTORY_FLAGS, dir_fd=parent_fd
                )
                try:
                    payload_fd = original_open(
                        "sentinel",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=replacement_fd,
                    )
                    try:
                        os.write(payload_fd, b"preserve replacement\n")
                    finally:
                        os.close(payload_fd)
                finally:
                    os.close(replacement_fd)
                replaced_directory = True
            return descriptor

        try:
            os.open = replace_after_directory_pin
            _expect_refusal(
                lambda: _create_owned_fixture_directory(parent_fd, _FIXTURE_NAME),
                "a replacement before the first directory stat",
            )
        finally:
            os.open = original_open
            os.close(parent_fd)
        if not replaced_directory:
            raise FixtureError("fixture replacement race self-test did not replace the path")
        with open(replacement_payload, "rb") as source:
            if source.read() != b"preserve replacement\n":
                raise FixtureError("fixture creation race changed its replacement sentinel")
        held_directory = os.path.join(
            replacement_root, "metal", _FIXTURE_NAME + ".held"
        )
        if not os.path.isdir(held_directory) or os.path.exists(
            os.path.join(held_directory, _OWNER_NAME)
        ):
            raise FixtureError("fixture creation race marked or removed the pinned directory")

        preopen_replacement_root = os.path.join(temporary, "preopen-replacement-build")
        os.makedirs(os.path.join(preopen_replacement_root, "metal"))
        parent_fd, _ = _open_build_parent(preopen_replacement_root)
        original_mkdir = os.mkdir
        preopen_replaced = False
        preopen_payload = os.path.join(
            preopen_replacement_root, "metal", _FIXTURE_NAME, "sentinel"
        )

        def replace_before_directory_open(path, mode=0o777, *, dir_fd=None):
            nonlocal preopen_replaced
            original_mkdir(path, mode, dir_fd=dir_fd)
            if (
                not preopen_replaced
                and path == _FIXTURE_NAME
                and dir_fd == parent_fd
            ):
                os.rename(
                    _FIXTURE_NAME,
                    _FIXTURE_NAME + ".held",
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                original_mkdir(_FIXTURE_NAME, mode, dir_fd=parent_fd)
                replacement_fd = original_open(
                    _FIXTURE_NAME, _DIRECTORY_FLAGS, dir_fd=parent_fd
                )
                try:
                    payload_fd = original_open(
                        "sentinel",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                        dir_fd=replacement_fd,
                    )
                    try:
                        os.write(payload_fd, b"preserve pre-open replacement\n")
                    finally:
                        os.close(payload_fd)
                finally:
                    os.close(replacement_fd)
                preopen_replaced = True

        try:
            os.mkdir = replace_before_directory_open
            _expect_refusal(
                lambda: _create_owned_fixture_directory(parent_fd, _FIXTURE_NAME),
                "a non-empty directory replacement before pinning",
            )
        finally:
            os.mkdir = original_mkdir
            os.close(parent_fd)
        if not preopen_replaced:
            raise FixtureError("pre-open replacement self-test did not replace the path")
        with open(preopen_payload, "rb") as source:
            if source.read() != b"preserve pre-open replacement\n":
                raise FixtureError("pre-open replacement sentinel was changed")
        if os.path.exists(
            os.path.join(preopen_replacement_root, "metal", _FIXTURE_NAME, _OWNER_NAME)
        ):
            raise FixtureError("pre-open replacement was incorrectly marked as owned")

        empty_replacement_root = os.path.join(temporary, "empty-replacement-build")
        os.makedirs(os.path.join(empty_replacement_root, "metal"))
        parent_fd, _ = _open_build_parent(empty_replacement_root)
        empty_replaced = False
        empty_replacement_payload = os.path.join(
            empty_replacement_root, "metal", _FIXTURE_NAME, "sentinel"
        )

        def replace_with_empty_directory(path, mode=0o777, *, dir_fd=None):
            nonlocal empty_replaced
            original_mkdir(path, mode, dir_fd=dir_fd)
            if (
                not empty_replaced
                and path == _FIXTURE_NAME
                and dir_fd == parent_fd
            ):
                os.rename(
                    _FIXTURE_NAME,
                    _FIXTURE_NAME + ".held",
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                original_mkdir(_FIXTURE_NAME, mode, dir_fd=parent_fd)
                empty_replaced = True

        try:
            os.mkdir = replace_with_empty_directory
            empty_fd, _empty_status = _create_owned_fixture_directory(
                parent_fd, _FIXTURE_NAME
            )
        finally:
            os.mkdir = original_mkdir
            os.close(parent_fd)
        os.close(empty_fd)
        if not empty_replaced:
            raise FixtureError("empty replacement self-test did not replace the path")
        with open(empty_replacement_payload, "wb") as output:
            output.write(b"preserve later data\n")
        prepared_fixture = _prepare(empty_replacement_root, fixture_tool)
        if os.path.basename(prepared_fixture) == _FIXTURE_NAME:
            raise FixtureError("MTP test fixture reused the empty replacement directory")
        with open(empty_replacement_payload, "rb") as source:
            if source.read() != b"preserve later data\n":
                raise FixtureError("a later fixture setup changed the empty replacement")

        marker_failure_root = os.path.join(temporary, "marker-failure-build")
        os.makedirs(os.path.join(marker_failure_root, "metal"))
        parent_fd, _ = _open_build_parent(marker_failure_root)
        original_open = os.open
        original_fstat = os.fstat
        marker_fds: list[int] = []
        failed_marker_stat = False

        def capture_marker_open(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == _OWNER_NAME and dir_fd is not None:
                marker_fds.append(descriptor)
            return descriptor

        def fail_first_marker_fstat(descriptor):
            nonlocal failed_marker_stat
            if marker_fds and descriptor == marker_fds[0] and not failed_marker_stat:
                failed_marker_stat = True
                raise OSError("simulated fstat failure after marker open")
            return original_fstat(descriptor)

        try:
            os.open = capture_marker_open
            os.fstat = fail_first_marker_fstat
            _expect_refusal(
                lambda: _create_owned_fixture_directory(parent_fd, _FIXTURE_NAME),
                "an ownership-marker stat failure after open",
            )
        finally:
            os.open = original_open
            os.fstat = original_fstat
            os.close(parent_fd)
        if not failed_marker_stat or len(marker_fds) != 1:
            raise FixtureError("ownership-marker post-open failure self-test did not run")
        try:
            original_fstat(marker_fds[0])
        except OSError:
            pass
        else:
            raise FixtureError("failed ownership-marker initialization leaked its descriptor")
        incomplete_marker = os.path.join(
            marker_failure_root, "metal", _FIXTURE_NAME, _OWNER_NAME
        )
        if not os.path.isfile(incomplete_marker) or os.path.getsize(incomplete_marker) != 0:
            raise FixtureError("marker initialization failure did not leave an inert partial marker")


def main(arguments: list[str]) -> int:
    if len(arguments) == 6 and arguments[0] == "--run-storage-test":
        try:
            output = _run_storage_test(*arguments[1:])
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                sys.stdout.write("\n")
            return 0
        except (FixtureError, OSError, subprocess.SubprocessError) as error:
            print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
            return 1
    if len(arguments) == 5 and arguments[0] == "--create-weight-cli-child":
        try:
            build_root, parent_dir, child_name, fixture_tool = arguments[1:]
            _run_self_tests(fixture_tool)
            _create_weight_cli_child(
                build_root, parent_dir, child_name, fixture_tool
            )
            return 0
        except (FixtureError, OSError, subprocess.SubprocessError) as error:
            print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
            return 1
    if len(arguments) == 2 and arguments[0] == "--self-test":
        try:
            _run_self_tests(arguments[1])
            return 0
        except (FixtureError, OSError, subprocess.SubprocessError) as error:
            print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
            return 1
    if len(arguments) != 3 or arguments[0] != "--prepare":
        print(
            "usage: prepare_mtp_storage_fixture.py --prepare BUILD_ROOT FIXTURE_TOOL "
            "| --run-storage-test BUILD_ROOT FIXTURE_TOOL STORAGE_TEST METALLIB "
            "ENGINE "
            "| --self-test FIXTURE_TOOL",
            file=sys.stderr,
        )
        return 2
    try:
        fixture_dir = _prepare(arguments[1], arguments[2])
        print(fixture_dir)
        return 0
    except (FixtureError, OSError, subprocess.SubprocessError) as error:
        print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
