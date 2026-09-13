"""一次性：model.py build_model 注册 v4（字节级，处理混合换行）。"""
from pathlib import Path

p = Path(r"E:\git\Luonnotar\glyphdet\core\model.py")
data = p.read_bytes()
anchor = b'def build_model(cfg):'
i = data.index(anchor)
# 找 anchor 所在行与后续 8 行的原始字节（保留其换行风格探测）
tail = data[i:i + 400]
print(repr(tail[:200]))
old = (
    b'def build_model(cfg):\r\n'
    b'    """\xe6\x8c\x89 cfg["model"]["arch"] \xe6\x9e\x84\xe9\x80\xa0\xe6\xa8\xa1\xe5\x9e\x8b\xef\xbc\x88\xe7\xbc\xba\xe7\x9c\x81 v1\xef\xbc\x8c\xe5\x90\x91\xe5\x90\x8e\xe5\x85\xbc\xe5\xae\xb9\xef\xbc\x89\xe3\x80\x82"""\r\n'
    b'    arch = cfg["model"].get("arch")\r\n'
    b'    if arch == "v3":\r\n'
    b'        return GlyphDetV3(cfg)\r\n'
    b'    if arch == "v2":\r\n'
    b'        return GlyphDetV2(cfg)\r\n'
    b'    return GlyphDet(cfg)\r\n'
)
new = old.replace(b'    if arch == "v3":',
                  b'    if arch == "v4":\r\n        return GlyphDetV4(cfg)\r\n    if arch == "v3":')
if old in data:
    p.write_bytes(data.replace(old, new, 1))
    print("PATCHED (CRLF variant)")
else:
    old_lf = old.replace(b"\r\n", b"\n")
    new_lf = new.replace(b"\r\n", b"\n")
    if old_lf in data:
        p.write_bytes(data.replace(old_lf, new_lf, 1))
        print("PATCHED (LF variant)")
    else:
        raise SystemExit("anchor bytes not found — 看上面 repr 手工对齐")
