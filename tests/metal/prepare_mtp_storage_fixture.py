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
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or status.st_mode & 0o077
        ):
            raise FixtureError(
                "MTP fixture ownership marker must be a private file owned by the current user"
            )
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


def _rollback_empty_fixture_directory(
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected_status: os.stat_result,
) -> None:
    """Roll back only a still-pinned new directory with no unexpected entries."""
    _verify_fixture_parent(parent_fd)
    opened_status = _verify_fixture_directory(directory_fd)
    if (opened_status.st_dev, opened_status.st_ino) != (
        expected_status.st_dev,
        expected_status.st_ino,
    ):
        raise FixtureError("new MTP fixture identity changed before rollback")
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (opened_status.st_dev, opened_status.st_ino):
        raise FixtureError("new MTP fixture path was replaced before rollback")

    entries = os.listdir(directory_fd)
    if entries == [_OWNER_NAME]:
        marker_fd = os.open(_OWNER_NAME, _FILE_FLAGS, dir_fd=directory_fd)
        try:
            marker_status = os.fstat(marker_fd)
            with os.fdopen(os.dup(marker_fd), "rb") as marker:
                content = marker.read(len(_OWNER_CONTENT) + 1)
            if (
                not stat.S_ISREG(marker_status.st_mode)
                or marker_status.st_uid != os.geteuid()
                or marker_status.st_mode & 0o077
                or len(content) > len(_OWNER_CONTENT)
                or not _OWNER_CONTENT.startswith(content)
            ):
                raise FixtureError("refusing to roll back an unrecognized marker")
            current_marker = os.stat(
                _OWNER_NAME, dir_fd=directory_fd, follow_symlinks=False
            )
            if (current_marker.st_dev, current_marker.st_ino) != (
                marker_status.st_dev,
                marker_status.st_ino,
            ):
                raise FixtureError("MTP fixture marker changed during rollback")
        finally:
            os.close(marker_fd)
        os.unlink(_OWNER_NAME, dir_fd=directory_fd)
    elif entries:
        raise FixtureError("refusing to roll back a fixture with unexpected data")

    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (opened_status.st_dev, opened_status.st_ino):
        raise FixtureError("new MTP fixture path changed during rollback")
    if os.listdir(directory_fd):
        raise FixtureError("new MTP fixture changed during rollback")
    os.rmdir(name, dir_fd=parent_fd)


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
    opened_status: os.stat_result | None = None
    identity_verified = False
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
        identity_verified = True
        _verify_empty_owned_fixture(directory_fd)
        _write_owner_marker(directory_fd)
        _verify_owner(directory_fd)
        _verify_only_owner_marker(directory_fd)
        return directory_fd, opened_status
    except BaseException as creation_error:
        rollback_error = None
        if identity_verified and opened_status is not None:
            try:
                _rollback_empty_fixture_directory(
                    parent_fd, name, directory_fd, opened_status
                )
            except Exception as error:
                rollback_error = error
        os.close(directory_fd)
        if rollback_error is not None:
            raise FixtureError(
                f"{creation_error}; MTP fixture rollback left the directory "
                f"for inspection: {rollback_error}"
            ) from creation_error
        raise


def _verify_created_fixture(directory_fd: int) -> None:
    _verify_fixture_directory(directory_fd)
    _verify_owner(directory_fd)
    if len(os.listdir(directory_fd)) <= 1:
        raise FixtureError("MTP fixture generator did not write the runtime files")


def _remove_fixture_contents(
    directory_fd: int, *, preserve_owner_marker: bool = False
) -> None:
    """Remove entries relative to pinned directory handles without following links."""
    for name in os.listdir(directory_fd):
        if preserve_owner_marker and name == _OWNER_NAME:
            # Keep the ownership proof until all other entries are removed.
            continue
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(entry.st_mode):
            child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                child_status = os.fstat(child_fd)
                if (child_status.st_dev, child_status.st_ino) != (
                    entry.st_dev,
                    entry.st_ino,
                ):
                    raise FixtureError(
                        "MTP fixture entry changed while opening it for cleanup"
                    )
                _remove_fixture_contents(child_fd)
                current = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) != (
                    child_status.st_dev,
                    child_status.st_ino,
                ):
                    raise FixtureError(
                        "MTP fixture directory changed during cleanup"
                    )
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            # unlink removes a symlink itself; it never follows its target.
            os.unlink(name, dir_fd=directory_fd)


def _remove_owned_fixture_directory(
    parent_fd: int,
    fixture_name: str,
    directory_fd: int,
    expected_status: os.stat_result,
    *,
    fixture_prefix: str = _FIXTURE_PREFIX,
) -> None:
    """Delete only the still-pinned, private fixture owned by this helper."""
    if (
        os.path.basename(fixture_name) != fixture_name
        or re.fullmatch(
            re.escape(fixture_prefix) + r"[0-9a-f]{32}", fixture_name
        )
        is None
    ):
        raise FixtureError("refusing to clean an invalid MTP fixture name")
    _verify_fixture_parent(parent_fd)
    opened_status = _verify_fixture_directory(directory_fd)
    _verify_owner(directory_fd)
    if (opened_status.st_dev, opened_status.st_ino) != (
        expected_status.st_dev,
        expected_status.st_ino,
    ):
        raise FixtureError("MTP fixture identity changed before cleanup")
    try:
        current = os.stat(
            fixture_name, dir_fd=parent_fd, follow_symlinks=False
        )
    except FileNotFoundError as error:
        raise FixtureError("MTP fixture path disappeared before cleanup") from error
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (opened_status.st_dev, opened_status.st_ino):
        raise FixtureError("MTP fixture path was replaced before cleanup")

    _remove_fixture_contents(directory_fd, preserve_owner_marker=True)
    current = os.stat(fixture_name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (
        current.st_dev,
        current.st_ino,
    ) != (opened_status.st_dev, opened_status.st_ino):
        raise FixtureError("MTP fixture path changed during cleanup")
    if os.listdir(directory_fd) != [_OWNER_NAME]:
        raise FixtureError("MTP fixture changed while being cleaned")
    _verify_owner(directory_fd)
    os.unlink(_OWNER_NAME, dir_fd=directory_fd)
    if os.listdir(directory_fd):
        raise FixtureError("MTP fixture changed before final removal")
    os.rmdir(fixture_name, dir_fd=parent_fd)


def _with_prepared_fixture(
    build_root: str, fixture_tool: str, action, *, cleanup_after_run: bool = False
):
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
                action_result = action(
                    directory_fd, directory_status, fixture_name
                )
            except BaseException as run_error:
                if cleanup_after_run:
                    try:
                        _remove_owned_fixture_directory(
                            parent_fd, fixture_name, directory_fd, directory_status
                        )
                    except Exception as cleanup_error:
                        raise FixtureError(
                            f"{run_error}; fixture cleanup also failed: {cleanup_error}"
                        ) from run_error
                raise
            if cleanup_after_run:
                _remove_owned_fixture_directory(
                    parent_fd, fixture_name, directory_fd, directory_status
                )
            return action_result
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

    return _with_prepared_fixture(
        build_root,
        fixture_tool,
        run_pinned_test,
        cleanup_after_run=True,
    )


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


def _open_weight_cli_test_output_parent(build_root: str) -> int:
    """Open build/metal/weight-store-cli without following path components."""
    root_input = os.path.normpath(os.path.abspath(build_root))
    resolved_root = os.path.realpath(root_input)
    if resolved_root != root_input:
        raise FixtureError("WeightStore CLI build root must not contain symlink components")
    root_fd = _absolute_directory_fd(resolved_root)
    metal_fd = -1
    cli_root_fd = -1
    keep_cli_root = False
    try:
        metal_fd = os.open("metal", _DIRECTORY_FLAGS, dir_fd=root_fd)
        cli_root_fd = os.open(
            "weight-store-cli", _DIRECTORY_FLAGS, dir_fd=metal_fd
        )
        _verify_fixture_parent(cli_root_fd)
        keep_cli_root = True
        return cli_root_fd
    except OSError as error:
        raise FixtureError(
            f"cannot securely open WeightStore CLI fixture output: {error}"
        ) from error
    finally:
        if cli_root_fd >= 0 and not keep_cli_root:
            os.close(cli_root_fd)
        if metal_fd >= 0:
            os.close(metal_fd)
        os.close(root_fd)


def _validate_weight_cli_test_name(fixture_name: str) -> None:
    if (
        os.path.basename(fixture_name) != fixture_name
        or re.fullmatch(r"fixtures-[0-9a-f]{32}", fixture_name) is None
    ):
        raise FixtureError("invalid WeightStore CLI test fixture name")


def _create_weight_cli_test_root(
    build_root: str, fixture_name: str
) -> tuple[int, int]:
    _validate_weight_cli_test_name(fixture_name)
    parent_fd = _open_weight_cli_test_output_parent(build_root)
    try:
        directory_fd, directory_status = _create_owned_fixture_directory(
            parent_fd, fixture_name
        )
        try:
            return directory_status.st_dev, directory_status.st_ino
        finally:
            os.close(directory_fd)
    finally:
        os.close(parent_fd)


def _remove_weight_cli_test_root(
    build_root: str, fixture_name: str, expected_device: int, expected_inode: int
) -> None:
    _validate_weight_cli_test_name(fixture_name)
    parent_fd = _open_weight_cli_test_output_parent(build_root)
    directory_fd = -1
    try:
        directory_fd = os.open(
            fixture_name, _DIRECTORY_FLAGS, dir_fd=parent_fd
        )
        directory_status = os.fstat(directory_fd)
        if (directory_status.st_dev, directory_status.st_ino) != (
            expected_device,
            expected_inode,
        ):
            raise FixtureError(
                "WeightStore CLI fixture was replaced before cleanup"
            )
        _remove_owned_fixture_directory(
            parent_fd,
            fixture_name,
            directory_fd,
            directory_status,
            fixture_prefix="fixtures-",
        )
    except OSError as error:
        raise FixtureError(
            f"cannot securely remove WeightStore CLI fixture: {error}"
        ) from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)


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

        cleanup_parent_fd, _ = _open_build_parent(root)
        cleanup_name = _FIXTURE_PREFIX + "a" * 32
        cleanup_fd, cleanup_status = _create_owned_fixture_directory(
            cleanup_parent_fd, cleanup_name
        )
        try:
            os.mkdir("nested", 0o700, dir_fd=cleanup_fd)
            nested_fd = os.open("nested", _DIRECTORY_FLAGS, dir_fd=cleanup_fd)
            try:
                payload_fd = os.open(
                    "payload",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=nested_fd,
                )
                try:
                    os.write(payload_fd, b"generated fixture\n")
                finally:
                    os.close(payload_fd)
            finally:
                os.close(nested_fd)
            os.symlink(sentinel, "external-link", dir_fd=cleanup_fd)
            _expect_refusal(
                lambda: _remove_owned_fixture_directory(
                    cleanup_parent_fd,
                    "../sentinel",
                    cleanup_fd,
                    cleanup_status,
                ),
                "a path-traversing fixture name during cleanup",
            )
            _remove_owned_fixture_directory(
                cleanup_parent_fd,
                cleanup_name,
                cleanup_fd,
                cleanup_status,
            )
            try:
                os.stat(cleanup_name, dir_fd=cleanup_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FixtureError("owned MTP fixture directory remained after cleanup")
        finally:
            os.close(cleanup_fd)
            os.close(cleanup_parent_fd)
        with open(sentinel, "rb") as source:
            if source.read() != b"keep\n":
                raise FixtureError("fixture cleanup followed a symlink outside its root")

        cleanup_lifecycle_root = os.path.join(temporary, "cleanup-lifecycle-build")
        os.makedirs(os.path.join(cleanup_lifecycle_root, "metal"))
        removed_name = _with_prepared_fixture(
            cleanup_lifecycle_root,
            fixture_tool,
            lambda _fd, _status, name: name,
            cleanup_after_run=True,
        )
        if os.path.lexists(
            os.path.join(cleanup_lifecycle_root, "metal", removed_name)
        ):
            raise FixtureError("successful fixture action did not clean its directory")

        failed_cleanup_root = os.path.join(temporary, "failed-cleanup-build")
        os.makedirs(os.path.join(failed_cleanup_root, "metal"))

        def fail_fixture_action(_fd, _status, _name):
            raise FixtureError("intentional cleanup lifecycle failure")

        _expect_refusal(
            lambda: _with_prepared_fixture(
                failed_cleanup_root,
                fixture_tool,
                fail_fixture_action,
                cleanup_after_run=True,
            ),
            "fixture action failure",
        )
        failed_names = os.listdir(os.path.join(failed_cleanup_root, "metal"))
        if failed_names:
            raise FixtureError("failed fixture action left its generated directory behind")

        combined_error_root = os.path.join(temporary, "combined-error-build")
        os.makedirs(os.path.join(combined_error_root, "metal"))
        original_remove_fixture = globals()["_remove_owned_fixture_directory"]

        def fail_fixture_cleanup(*_args, **_kwargs):
            raise RuntimeError("simulated cleanup exception")

        globals()["_remove_owned_fixture_directory"] = fail_fixture_cleanup
        try:
            try:
                _with_prepared_fixture(
                    combined_error_root,
                    fixture_tool,
                    fail_fixture_action,
                    cleanup_after_run=True,
                )
            except FixtureError as error:
                if (
                    "intentional cleanup lifecycle failure" not in str(error)
                    or "simulated cleanup exception" not in str(error)
                ):
                    raise FixtureError(
                        "combined fixture failure report omitted the action or cleanup error"
                    ) from error
            else:
                raise FixtureError("combined fixture failure self-test did not fail")
        finally:
            globals()["_remove_owned_fixture_directory"] = original_remove_fixture

        replacement_name = _FIXTURE_PREFIX + "b" * 32
        replacement_parent_fd, _ = _open_build_parent(root)
        replacement_fd, replacement_status = _create_owned_fixture_directory(
            replacement_parent_fd, replacement_name
        )
        replacement_path = os.path.join(parent, replacement_name)
        replacement_payload = os.path.join(replacement_path, "sentinel")
        try:
            os.rename(
                replacement_name,
                replacement_name + ".held",
                src_dir_fd=replacement_parent_fd,
                dst_dir_fd=replacement_parent_fd,
            )
            os.mkdir(replacement_name, 0o700, dir_fd=replacement_parent_fd)
            replacement_path_fd = os.open(
                replacement_name,
                _DIRECTORY_FLAGS,
                dir_fd=replacement_parent_fd,
            )
            try:
                payload_fd = os.open(
                    "sentinel",
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=replacement_path_fd,
                )
                try:
                    os.write(payload_fd, b"preserve replacement\n")
                finally:
                    os.close(payload_fd)
            finally:
                os.close(replacement_path_fd)
            _expect_refusal(
                lambda: _remove_owned_fixture_directory(
                    replacement_parent_fd,
                    replacement_name,
                    replacement_fd,
                    replacement_status,
                ),
                "a replaced fixture path during cleanup",
            )
        finally:
            os.close(replacement_fd)
            os.close(replacement_parent_fd)
        with open(replacement_payload, "rb") as source:
            if source.read() != b"preserve replacement\n":
                raise FixtureError("fixture cleanup changed a replacement sentinel")

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
        if not failed_marker_stat or len(marker_fds) != 2:
            raise FixtureError("ownership-marker post-open failure self-test did not run")
        for marker_fd in marker_fds:
            try:
                original_fstat(marker_fd)
            except OSError:
                pass
            else:
                raise FixtureError(
                    "failed ownership-marker initialization leaked its descriptor"
                )
        incomplete_marker = os.path.join(
            marker_failure_root, "metal", _FIXTURE_NAME, _OWNER_NAME
        )
        if os.path.lexists(incomplete_marker) or os.path.lexists(
            os.path.dirname(incomplete_marker)
        ):
            raise FixtureError("owned partial fixture was not rolled back safely")

        partial_marker_root = os.path.join(temporary, "partial-marker-build")
        os.makedirs(os.path.join(partial_marker_root, "metal"))
        partial_parent_fd, _ = _open_build_parent(partial_marker_root)
        original_open = os.open
        original_write = os.write
        partial_marker_fds: list[int] = []
        partial_marker_written = False

        def capture_partial_marker_open(path, flags, mode=0o777, *, dir_fd=None):
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == _OWNER_NAME and dir_fd is not None:
                partial_marker_fds.append(descriptor)
            return descriptor

        def fail_after_partial_marker_write(descriptor, data):
            nonlocal partial_marker_written
            if (
                partial_marker_fds
                and descriptor == partial_marker_fds[0]
                and not partial_marker_written
            ):
                prefix = data[:8]
                if original_write(descriptor, prefix) != len(prefix):
                    raise FixtureError("partial marker write self-test was short")
                partial_marker_written = True
                raise OSError("simulated failure after partial marker write")
            return original_write(descriptor, data)

        try:
            os.open = capture_partial_marker_open
            os.write = fail_after_partial_marker_write
            _expect_refusal(
                lambda: _create_owned_fixture_directory(
                    partial_parent_fd, _FIXTURE_NAME
                ),
                "an ownership-marker write failure after a valid prefix",
            )
        finally:
            os.open = original_open
            os.write = original_write
            os.close(partial_parent_fd)
        if not partial_marker_written or len(partial_marker_fds) != 2:
            raise FixtureError("partial ownership-marker write self-test did not run")
        for marker_fd in partial_marker_fds:
            try:
                original_fstat(marker_fd)
            except OSError:
                pass
            else:
                raise FixtureError("partial-marker rollback leaked a descriptor")
        partial_marker_path = os.path.join(
            partial_marker_root, "metal", _FIXTURE_NAME
        )
        if os.path.lexists(partial_marker_path):
            raise FixtureError("a valid partial marker fixture was not rolled back")

        unrecognized_marker_root = os.path.join(
            temporary, "unrecognized-marker-build"
        )
        os.makedirs(os.path.join(unrecognized_marker_root, "metal"))
        unrecognized_parent_fd, _ = _open_build_parent(unrecognized_marker_root)
        unrecognized_fd, unrecognized_status = _create_owned_fixture_directory(
            unrecognized_parent_fd, _FIXTURE_NAME
        )
        try:
            marker_fd = os.open(
                _OWNER_NAME,
                os.O_WRONLY | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=unrecognized_fd,
            )
            try:
                os.write(marker_fd, b"unrecognized marker")
            finally:
                os.close(marker_fd)
            _expect_refusal(
                lambda: _rollback_empty_fixture_directory(
                    unrecognized_parent_fd,
                    _FIXTURE_NAME,
                    unrecognized_fd,
                    unrecognized_status,
                ),
                "a non-prefix ownership marker",
            )
            with open(
                os.path.join(
                    unrecognized_marker_root, "metal", _FIXTURE_NAME, _OWNER_NAME
                ),
                "rb",
            ) as marker:
                if marker.read() != b"unrecognized marker":
                    raise FixtureError(
                        "rollback changed an unrecognized ownership marker"
                    )
        finally:
            os.close(unrecognized_fd)
            os.close(unrecognized_parent_fd)


def main(arguments: list[str]) -> int:
    if len(arguments) == 3 and arguments[0] == "--create-weight-cli-test-root":
        try:
            device, inode = _create_weight_cli_test_root(
                arguments[1], arguments[2]
            )
            print(f"{device} {inode}")
            return 0
        except (FixtureError, OSError, subprocess.SubprocessError) as error:
            print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
            return 1
    if len(arguments) == 5 and arguments[0] == "--remove-weight-cli-test-root":
        try:
            _remove_weight_cli_test_root(
                arguments[1],
                arguments[2],
                int(arguments[3]),
                int(arguments[4]),
            )
            return 0
        except (FixtureError, OSError, subprocess.SubprocessError) as error:
            print(f"prepare_mtp_storage_fixture: {error}", file=sys.stderr)
            return 1
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
            "| --create-weight-cli-test-root BUILD_ROOT FIXTURE_NAME "
            "| --remove-weight-cli-test-root BUILD_ROOT FIXTURE_NAME DEVICE INODE "
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
