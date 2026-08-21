"""从仓库根目录的 LICENSE 生成 desktop/packaging/license.rtf。

license.rtf 是 WiX（.msi）许可页 `WixUILicenseRtf` 读取的文件。手写一份 RTF 扔在
仓库里、改了 LICENSE 却忘了同步，是这类文件最常见的失效方式，所以这里用脚本生成。

两个必须守住的点：

1. **输出必须是纯 ASCII。** Windows Installer 的许可页用的是 RichEdit 控件，
   直接塞 UTF-8 字节会整片变成乱码。所有非 ASCII 字符一律写成 RTF 的 ``\\uN?``
   转义（N 是有符号 16 位十进制，超过 32767 的码位要减 65536）。
2. **段落用 ``\\par`` 重排。** LICENSE 里的换行是源码硬折行，原样搬过去会在
   许可页里折成锯齿状。这里按空行切段、段内合并空白，让控件自己换行。

用法（改完 LICENSE 之后跑一次）::

    python desktop/packaging/make_rtf.py
"""

from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent          # desktop/packaging
REPO_ROOT = HERE.parent.parent                          # 仓库根目录
LICENSE_FILE = REPO_ROOT / "LICENSE"
OUTPUT_FILE = HERE / "license.rtf"

TITLE_ZH = "许可与用途声明"

NOTICE_ZH = [
    "本软件为研究用决策支持工具，不是筛查工具、不是诊断工具、不产生医嘱，"
    "也不替代医师或药师作出治疗与用药决定。所有结果必须由具备资质的临床或"
    "药学人员结合完整病历与随访结果解释后方可使用。",
    "全部分析在本机完成，不向外部服务器上传输入数据。请勿录入超出研究伦理"
    "批准范围的可识别患者信息。",
]

NOTICE_EN = (
    "This software is a research-use decision-support tool. It is not a screening "
    "device, not a diagnostic device, and does not produce medical orders. It does "
    "not replace the judgement of a qualified physician or pharmacist. All output "
    "must be interpreted by qualified clinical or pharmacy staff together with the "
    "complete medical record. All analysis runs locally; no input data is uploaded."
)


def rtf_escape(text: str) -> str:
    """把任意文本转成 RTF 正文可直接嵌入的纯 ASCII 片段。"""
    out: list[str] = []
    for ch in text:
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ord(ch) < 128:
            out.append(ch)
        else:
            code = ord(ch)
            if code > 32767:          # RTF 的 \uN 是有符号 16 位
                code -= 65536
            out.append("\\u%d?" % code)
    return "".join(out)


def build_rtf(license_text: str) -> str:
    lines = [
        r"{\rtf1\ansi\ansicpg936\deff0\deflang2052",
        r"{\fonttbl{\f0\fnil\fcharset0 Segoe UI;}"
        r"{\f1\fnil\fcharset134 Microsoft YaHei;}}",
        r"\viewkind4\uc1\pard\sa120\f1\fs18",
        r"\b " + rtf_escape(TITLE_ZH) + r"\b0\par",
    ]
    for paragraph in NOTICE_ZH:
        lines.append(rtf_escape(paragraph) + r"\par")

    lines.append(r"\pard\sa120\f0\fs18 " + rtf_escape(NOTICE_EN) + r"\par")
    lines.append(r"\pard\sa120\f0\fs18\b MIT License\b0\par")

    body = license_text.replace("MIT License\n", "", 1)
    for paragraph in (p.strip() for p in body.split("\n\n")):
        if paragraph:
            lines.append(rtf_escape(" ".join(paragraph.split())) + r"\par")

    lines.append("}")
    return "\r\n".join(lines) + "\r\n"


def main() -> None:
    if not LICENSE_FILE.is_file():
        raise SystemExit("找不到 LICENSE：%s" % LICENSE_FILE)

    rtf = build_rtf(LICENSE_FILE.read_text(encoding="utf-8"))
    if not rtf.isascii():
        raise SystemExit("生成结果含非 ASCII 字符，转义逻辑有问题")

    OUTPUT_FILE.write_bytes(rtf.encode("ascii"))
    print("wrote %s (%d bytes)" % (OUTPUT_FILE, len(rtf)))


if __name__ == "__main__":
    main()
