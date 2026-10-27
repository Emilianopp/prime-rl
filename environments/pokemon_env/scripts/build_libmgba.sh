#!/bin/bash
# Set up the emulator side of pokemon-env:
#   - EMU_VENV: a Python 3.11 venv with numpy (the interpreter that runs the games;
#     mGBA's bindings hang on Python >= 3.12 — see pokemon_env/firered/worker.py)
#   - SITE: the PyPI `mgba` 0.10.2 bindings plus the libmgba they were compiled against,
#     built from source here (the wheels ship _pylib without libmgba)
# Run on a compute node, then point the env at them:
#
#   sbatch -p main-cpu -c 8 --mem=16G -t 1:00:00 --wrap "bash environments/pokemon_env/scripts/build_libmgba.sh"
#   export POKEMON_EMULATOR_PYTHON=$EMU_VENV/bin/python POKEMON_EMULATOR_PYTHONPATH=$SITE
set -euo pipefail

ROOT=${ROOT:-$SCRATCH/pokemon}
SITE=${SITE:-$ROOT/py-mgba}
EMU_VENV=${EMU_VENV:-$ROOT/emulator-venv}
TOOLS=$EMU_VENV
SRC=$ROOT/mgba-src
export UV_CACHE_DIR=/tmp/uvcache_$USER UV_LINK_MODE=copy  # hard links on $SCRATCH load as "Bad message"

echo "== tools"; which gcc g++ || true; gcc --version | head -1
[ -x "$TOOLS/bin/python" ] || ~/.local/bin/uv venv -q --seed --python 3.11 "$TOOLS"
VIRTUAL_ENV=$TOOLS ~/.local/bin/uv pip install -q "cmake<4" patchelf numpy cffi
export PATH=$TOOLS/bin:$PATH

if [ ! -f "$SITE/mgba/__init__.py" ]; then
  WHEELS=$(mktemp -d)
  "$TOOLS/bin/python" -m pip download -q --no-deps --only-binary=:all: --python-version 3.11 \
    --platform manylinux1_x86_64 -d "$WHEELS" mgba==0.10.2 cached-property
  mkdir -p "$SITE"
  for w in "$WHEELS"/*.whl; do "$TOOLS/bin/python" -m zipfile -e "$w" "$SITE"; done
fi
PYLIB=$(ls "$SITE"/mgba/_pylib*.so)
cmake --version | head -1
for h in zlib.h png.h; do test -f /usr/include/$h && echo "system $h: yes" || echo "system $h: no"; done

if [ ! -d $SRC/.git ]; then
  git clone -q --depth 1 --branch 0.10.2 https://github.com/mgba-emu/mgba.git $SRC
fi
cd $SRC

# libpng from mGBA's bundled copy (static, PIC); zlib from the system or bundled copy.
ZLIB_ARGS=()
if [ ! -f /usr/include/zlib.h ]; then
  cmake -S src/third-party/zlib -B build-zlib -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON >/dev/null
  cmake --build build-zlib -j"$(nproc)" >/dev/null
  mkdir -p zlib-inc && cp src/third-party/zlib/zlib.h build-zlib/zconf.h zlib-inc/
  ZLIB_ARGS=(-DZLIB_INCLUDE_DIR=$SRC/zlib-inc -DZLIB_LIBRARY=$SRC/build-zlib/libz.a)
fi
cmake -S src/third-party/libpng -B build-png -DCMAKE_BUILD_TYPE=Release -DPNG_SHARED=OFF -DPNG_STATIC=ON \
  -DPNG_TESTS=OFF -DPNG_EXECUTABLES=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON "${ZLIB_ARGS[@]}" >/dev/null
cmake --build build-png -j"$(nproc)" >/dev/null
mkdir -p png-inc && cp src/third-party/libpng/png.h src/third-party/libpng/pngconf.h build-png/pnglibconf.h png-inc/

# The bindings reference e-Reader card-scanning symbols that only exist in
# FFmpeg-enabled builds, plus ELF loading (libelf); nothing here uses them, so compile no-op stubs.
nm -D --undefined-only "$PYLIB" | awk '{print $2}' | grep -E '^(EReader|mCoreLoadELF)' | sort -u > /tmp/ereader_syms_$USER.txt
{ echo "/* No-op stubs for features this build omits (e-Reader scanning needs FFmpeg, ELF loading needs libelf). */"
  while read -r s; do echo "void $s(void) {}"; done < /tmp/ereader_syms_$USER.txt; } > src/gba/cart/ereader_stubs.c
grep -q ereader_stubs.c src/gba/CMakeLists.txt || sed -i 's#\tcart/ereader.c#\tcart/ereader.c\n\tcart/ereader_stubs.c#' src/gba/CMakeLists.txt
echo "stubbed symbols: $(wc -l < /tmp/ereader_syms_$USER.txt)"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED=ON -DBUILD_STATIC=OFF -DBUILD_QT=OFF -DBUILD_SDL=OFF \
  -DBUILD_GL=OFF -DBUILD_GLES2=OFF -DBUILD_GLES3=OFF -DUSE_FFMPEG=OFF -DUSE_EPOXY=OFF -DUSE_LUA=OFF \
  -DUSE_DISCORD_RPC=OFF -DUSE_EDITLINE=OFF -DUSE_LIBZIP=OFF -DUSE_MINIZIP=OFF -DUSE_ELF=OFF -DUSE_SQLITE3=OFF \
  -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DBUILD_SUITE=OFF -DUSE_PNG=ON \
  -DPNG_LIBRARY_RELEASE=$SRC/build-png/libpng.a -DPNG_PNG_INCLUDE_DIR=$SRC/png-inc "${ZLIB_ARGS[@]}" >/dev/null
cmake --build build -j"$(nproc)" 2>&1 | grep -E " error|fatal" || true

LIB=$(ls build/libmgba.so.0.10.* | head -1)
cp "$LIB" "$SITE/mgba/libmgba.so.0.10"
patchelf --set-rpath '$ORIGIN' "$PYLIB"

echo "== unresolved symbols (should be empty)"
nm -D --undefined-only "$PYLIB" | awk '{print $2}' | sort -u > /tmp/need_$USER.txt
nm -D --defined-only "$SITE/mgba/libmgba.so.0.10" | awk '{print $3}' | sort -u > /tmp/have_$USER.txt
comm -23 /tmp/need_$USER.txt /tmp/have_$USER.txt | grep -vE '^(Py|_Py|_)|@' || true
PYTHONPATH=$SITE "$TOOLS/bin/python" -c "import mgba.core; print('import mgba: ok')"
echo "export POKEMON_EMULATOR_PYTHON=$TOOLS/bin/python POKEMON_EMULATOR_PYTHONPATH=$SITE"
