PYTHON ?= python3
SWIG ?= swig

.PHONY: all swig build clean

all: build

# Generate SWIG wrapper -> build/micropolisengine_wrap.cpp and build/micropolisengine.py
swig: build/micropolisengine_wrap.cpp

build/micropolisengine_wrap.cpp: swig/micropolisengine.i swig/micropolisengine-swig-python.i src/micropolis.h
	mkdir -p build
	$(SWIG) -python -py3 -c++ -Isrc -Wall \
	        -outdir build \
	        -o build/micropolisengine_wrap.cpp \
	        swig/micropolisengine.i

# Build the Python extension in-place (lands as _micropolisengine*.so next to micropolisengine.py)
build: swig
	$(PYTHON) setup.py build_ext --inplace --build-temp build/temp --build-lib build

clean:
	rm -rf build *.egg-info __pycache__
