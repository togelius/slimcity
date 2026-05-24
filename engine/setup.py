from setuptools import setup, Extension

sources = [
    "build/micropolisengine_wrap.cpp",
    "src/allocate.cpp",
    "src/animate.cpp",
    "src/budget.cpp",
    "src/connect.cpp",
    "src/disasters.cpp",
    "src/evaluate.cpp",
    "src/fileio.cpp",
    "src/generate.cpp",
    "src/graph.cpp",
    "src/initialize.cpp",
    "src/main.cpp",
    "src/map.cpp",
    "src/message.cpp",
    "src/micropolis.cpp",
    "src/position.cpp",
    "src/power.cpp",
    "src/random.cpp",
    "src/resource.cpp",
    "src/scan.cpp",
    "src/simulate.cpp",
    "src/sprite.cpp",
    "src/stubs.cpp",
    "src/tool.cpp",
    "src/traffic.cpp",
    "src/update.cpp",
    "src/utilities.cpp",
    "src/zone.cpp",
]

ext = Extension(
    "_micropolisengine",
    sources=sources,
    include_dirs=["src"],
    extra_compile_args=["-std=c++11", "-Wno-deprecated-declarations"],
    language="c++",
)

setup(
    name="micropolisengine",
    version="0.1.0",
    ext_modules=[ext],
    py_modules=["micropolisengine"],
    package_dir={"": "build"},
)
