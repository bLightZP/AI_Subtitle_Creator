# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("faster_whisper", includes=["assets/*"])
datas += [("assets\\app_icon.ico", "assets"), ("assets\\app_icon.png", "assets")]
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
hiddenimports = [
    "av",
    "ctranslate2",
    "faster_whisper",
    "huggingface_hub",
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "tokenizers",
]
excludes = [
    "IPython",
    "_pytest",
    "fsspec.gui",
    "matplotlib",
    "notebook",
    "onnx",
    "onnxruntime.quantization",
    "onnxruntime.tools",
    "onnxruntime.transformers",
    "pandas",
    "pytest",
    "scipy",
    "tensorflow",
    "tokenizers.tools",
    "torch",
]

a = Analysis(
    ["src\\ai_subtitle_creator\\cli.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="aisub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets\\app_icon.ico",
)

