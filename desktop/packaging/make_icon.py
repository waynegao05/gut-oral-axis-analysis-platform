"""从 app-icon-source.png 生成桌面端的 AppIcon.ico / AppIcon.png。

源图是白底的双色标志。这里做三件事：

1. **去白底，而不是抠白色。** 白底图的每个像素其实是 ``p = a*ink + (1-a)*255``。
   先聚类出画面真正用到的两种墨色（深蓝灰、青），再把向量 ``255-p`` 投影到
   ``255-ink`` 方向上：投影长度就是该像素的覆盖率 a，残差最小的那种墨色就是它
   本来的颜色。直接按亮度阈值抠图会把抗锯齿边缘啃出白毛边，这个做法不会。

2. **反白 + 深色圆角底板。** 原标志是深蓝灰的，放在 Windows 11 深色任务栏
   （约 #202020）上几乎隐形。所以线条反白、青色提亮，垫一块深色圆角底板，
   浅色和深色主题下都清楚。

3. **多尺寸 .ico。** 16 到 256 px 全套。小尺寸下细节必然糊，所以尺寸越小
   留白越少（见 INSET_BY_SIZE），让图形本身占满可用像素。

改了源图之后重跑一次（然后重新构建桌面端与安装器）::

    python desktop/packaging/make_icon.py

产物只有一份，放在应用自己的 Assets 目录下，四处引用它，避免多份副本走样：

* ``GutOralAxis.Desktop.csproj`` 的 ``ApplicationIcon``  -> 嵌进 exe，任务栏/资源管理器用
* ``MainWindow.xaml.cs`` 运行时加载 ``AppIcon.ico``      -> 窗口图标
* ``MainWindow.xaml`` 标题栏里的 ``AppIcon.png``
* ``installer.iss`` 的 ``SetupIconFile`` 与 ``GutOralAxisDesktop.wxs`` 的 ``ARPPRODUCTICON``
"""

from __future__ import annotations

import io
import pathlib
import struct

import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent                    # desktop/packaging
ASSETS = HERE.parent / "src" / "GutOralAxis.Desktop" / "Assets"
SOURCE_FILE = HERE / "app-icon-source.png"
ICON_FILE = ASSETS / "AppIcon.ico"
PNG_FILE = ASSETS / "AppIcon.png"
PNG_SIZE = 512                   # 标题栏那张按 27px 显示，512 够用且和原资源一致

# 反白后的两种墨色，以及底板色
INK_LIGHT = (240, 245, 248)      # 原深蓝灰线条 -> 近白
INK_ACCENT = (86, 199, 189)      # 原青色 -> 提亮的青
PLATE = (33, 41, 54)             # 圆角底板
PLATE_RADIUS = 0.22              # 圆角半径，占边长比例

MASTER = 1024                    # 先渲染到这个尺寸再逐级缩小
ICO_SIZES = [256, 128, 64, 48, 40, 32, 24, 20, 16]
# 尺寸越小留白越少，否则 16px 下图形只剩十几个像素
INSET_BY_SIZE = {256: 0.16, 128: 0.16, 64: 0.14, 48: 0.12, 40: 0.11,
                 32: 0.09, 24: 0.07, 20: 0.06, 16: 0.05}

# 覆盖率低于此值一律当成背景。源图白底并非纯 255（实测 253~255），
# 不设这个下限的话整张图会蒙上一层 1~3 的噪声 alpha。
COVERAGE_FLOOR = 0.05


def separate_inks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (coverage, ink_index)：每像素的覆盖率与它属于哪种墨色。"""
    lum = rgb.mean(2)
    core = rgb[lum < 140].reshape(-1, 3)
    if len(core) == 0:
        raise SystemExit("源图里找不到深色像素，确认它是白底彩色标志")

    # 2-means，用蓝/青两端做种子，避免随机初始化导致结果不稳定
    seeds = np.array([[40.0, 45.0, 62.0], [62.0, 130.0, 130.0]])
    for _ in range(40):
        label = ((core[:, None, :] - seeds[None]) ** 2).sum(2).argmin(1)
        for k in range(2):
            if (label == k).any():
                seeds[k] = core[label == k].mean(0)

    height, width, _ = rgb.shape
    delta = 255.0 - rgb
    coverage = np.zeros((height, width))
    residual = np.full((height, width), np.inf)
    ink = np.zeros((height, width), dtype=int)

    for k, seed in enumerate(seeds):
        direction = 255.0 - seed
        t = (delta @ direction) / (direction @ direction)
        r = np.linalg.norm(delta - t[..., None] * direction, axis=2)
        closer = r < residual
        residual[closer] = r[closer]
        coverage[closer] = t[closer]
        ink[closer] = k

    coverage = np.clip(coverage, 0.0, 1.0)
    coverage[coverage < COVERAGE_FLOOR] = 0.0
    return coverage, ink


def build_glyph(coverage: np.ndarray, ink: np.ndarray) -> Image.Image:
    """反白上色，裁到内容边界，补成正方形。"""
    recoloured = np.zeros(coverage.shape + (3,))
    recoloured[ink == 0] = INK_LIGHT
    recoloured[ink == 1] = INK_ACCENT
    recoloured[coverage == 0] = 0

    rgba = np.dstack([recoloured, coverage * 255.0]).round().astype(np.uint8)
    glyph = Image.fromarray(rgba, "RGBA")

    box = glyph.getbbox()
    if box is None:
        raise SystemExit("去底之后什么都不剩，检查 COVERAGE_FLOOR")
    glyph = glyph.crop(box)

    side = max(glyph.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2))
    return square.resize((MASTER, MASTER), Image.LANCZOS)


def compose(glyph: Image.Image, size: int, inset: float) -> Image.Image:
    plate = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    mask = Image.new("L", (MASTER, MASTER), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, MASTER - 1, MASTER - 1], int(MASTER * PLATE_RADIUS), fill=255)
    plate.paste(Image.new("RGBA", (MASTER, MASTER), PLATE + (255,)), (0, 0), mask)

    inner = int(MASTER * (1 - 2 * inset))
    scaled = glyph.resize((inner, inner), Image.LANCZOS)
    plate.paste(scaled, ((MASTER - inner) // 2, (MASTER - inner) // 2), scaled)
    return plate.resize((size, size), Image.LANCZOS)


def encode_bmp_frame(image: Image.Image) -> bytes:
    """把一帧编码成 .ico 内部的 BMP（DIB）表示。

    Pillow 的 ICO 写出器会把所有帧都存成 PNG。PNG 帧是 Vista 才引入的，
    历史上只在 256px 这一档被广泛依赖；小尺寸仍以 BMP 兼容性最好，一些工具链
    （含部分资源编辑器和图标读取路径）碰到全 PNG 的 .ico 会读不出小图。
    所以这里自己写：<256 用 BMP，256 用 PNG。

    ICO 里的 BMP 有两个坑：位图高度要写成实际高度的两倍（XOR 位图 + AND 掩码
    各占一半），像素按自下而上、BGRA 存放。32 位图标的 AND 掩码其实用不上，
    但结构上必须存在，全写 0 即可。
    """
    width, height = image.size
    bgra = np.array(image.convert("RGBA"))[::-1]          # 自下而上
    bgra = bgra[..., [2, 1, 0, 3]]                        # RGBA -> BGRA

    header = struct.pack(
        "<IiiHHIIiiII",
        40,            # biSize
        width,
        height * 2,    # biHeight：XOR + AND
        1,             # biPlanes
        32,            # biBitCount
        0,             # biCompression = BI_RGB
        0,             # biSizeImage
        0, 0,          # 分辨率，图标里无意义
        0, 0,          # 调色板
    )
    mask_stride = ((width + 31) // 32) * 4                # 1bpp，行按 4 字节对齐
    return header + bgra.tobytes() + b"\x00" * (mask_stride * height)


def write_ico(frames: list[Image.Image], path: pathlib.Path) -> None:
    payloads = []
    for frame in frames:
        if frame.width >= 256:
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG", optimize=True)
            payloads.append(buffer.getvalue())
        else:
            payloads.append(encode_bmp_frame(frame))

    offset = 6 + 16 * len(frames)
    directory = struct.pack("<HHH", 0, 1, len(frames))
    for frame, payload in zip(frames, payloads):
        directory += struct.pack(
            "<BBBBHHII",
            frame.width % 256,      # 256 在目录项里记作 0
            frame.height % 256,
            0,                      # 调色板色数，32 位图标为 0
            0,                      # 保留
            1,                      # 色彩平面
            32,                     # 位深
            len(payload),
            offset,
        )
        offset += len(payload)

    path.write_bytes(directory + b"".join(payloads))


def main() -> None:
    if not SOURCE_FILE.is_file():
        raise SystemExit("找不到源图：%s" % SOURCE_FILE)
    if not ASSETS.is_dir():
        raise SystemExit("找不到 Assets 目录：%s" % ASSETS)

    rgb = np.array(Image.open(SOURCE_FILE).convert("RGB")).astype(np.float64)
    coverage, ink = separate_inks(rgb)
    glyph = build_glyph(coverage, ink)

    # 每一档都从 1024 的母版单独 LANCZOS 缩下来，而不是逐级递缩，
    # 免得小尺寸叠加多次重采样的模糊
    frames = [compose(glyph, size, INSET_BY_SIZE[size]) for size in ICO_SIZES]
    write_ico(frames, ICON_FILE)
    compose(glyph, PNG_SIZE, INSET_BY_SIZE[256]).save(PNG_FILE)

    print("wrote %s (%d bytes, %d sizes: %s)"
          % (ICON_FILE, ICON_FILE.stat().st_size, len(ICO_SIZES),
             ", ".join(str(s) for s in ICO_SIZES)))
    print("wrote %s (%dx%d)" % (PNG_FILE, PNG_SIZE, PNG_SIZE))


if __name__ == "__main__":
    main()
