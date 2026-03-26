"""Minimal multipart/form-data parser. Replaces deprecated cgi.FieldStorage."""
from __future__ import annotations

import io


def parse_content_type(header: str) -> tuple[str, dict[str, str]]:
    """Parse Content-Type or Content-Disposition header; return (main type, params dict)."""
    parts = header.split(";")
    ctype = parts[0].strip().lower()
    params = {}
    for p in parts[1:]:
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        params[k.strip().lower()] = v.strip().strip('"')
    return ctype, params


def parse_multipart(fp: io.BufferedIOBase, boundary: str) -> dict[str, "Part"]:
    """
    Parse a multipart/form-data body.
    Returns dict mapping field name -> Part.
    Caller should limit input size (e.g. cap Content-Length) before reading.
    """
    sep = ("--" + boundary).encode()
    end = ("--" + boundary + "--").encode()
    raw = fp.read()
    parts_raw = raw.split(sep)
    result: dict[str, Part] = {}

    for chunk in parts_raw:
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--" or chunk.startswith(end):
            continue

        if b"\r\n\r\n" in chunk:
            header_block, body = chunk.split(b"\r\n\r\n", 1)
        elif b"\n\n" in chunk:
            header_block, body = chunk.split(b"\n\n", 1)
        else:
            continue

        if body.endswith(b"\r\n"):
            body = body[:-2]

        headers: dict[str, str] = {}
        for line in header_block.decode("utf-8", errors="replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        disp = headers.get("content-disposition", "")
        _, disp_params = parse_content_type(disp)
        name = disp_params.get("name", "")
        filename = disp_params.get("filename")

        result[name] = Part(name=name, filename=filename, data=body, headers=headers)

    return result


class Part:
    __slots__ = ("name", "filename", "data", "headers")

    def __init__(self, name: str, filename: str | None, data: bytes, headers: dict[str, str]):
        self.name = name
        self.filename = filename
        self.data = data
        self.headers = headers

    @property
    def is_file(self) -> bool:
        return self.filename is not None

    @property
    def value(self) -> str:
        return self.data.decode("utf-8", errors="replace")
