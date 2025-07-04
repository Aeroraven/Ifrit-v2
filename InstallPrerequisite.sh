# Tbb build_thirdparty/tbb/lib/cmake/TBB, check path
current_dir=$(pwd)
echo "Current directory: $current_dir"
tbb_config_path="./build_thirdparty/tbb/lib/cmake/TBB/TBBConfig.cmake"
mkdir -p "$current_dir/build_thirdparty/"

install_base="$current_dir/build_thirdparty"
install_base_tbb="$install_base/tbb"

if [[ -f "$tbb_config_path" ]]; then
  echo "TBB found, skipping installation"
else
  echo "TBB not found, installing"
  mkdir -p "$install_base_tbb"
  cmake -S ./dependencies/tbb -B ./dependencies/tbb/build -DCMAKE_INSTALL_PREFIX="$install_base_tbb" -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF
  cmake --build ./dependencies/tbb/build --config Release
  cmake --install ./dependencies/tbb/build
fi

# Blosc build_thirdparty/blosc/lib/cmake/Blosc, check path
blosc_config_path="./build_thirdparty/blosc/lib/libblosc.lib"
install_base_blosc="$install_base/blosc"
if [[ -f "$blosc_config_path" ]]; then
  echo "Blosc found, skipping installation"
else
  echo "Blosc not found, installing"
  mkdir -p "$install_base_blosc"
  cmake -S ./dependencies/blosc -B ./dependencies/blosc/build -DCMAKE_INSTALL_PREFIX="$install_base_blosc" -DCMAKE_BUILD_TYPE=Release
  cmake --build ./dependencies/blosc/build --config Release
  cmake --install ./dependencies/blosc/build
fi

# zlib build_thirdparty/zlib/lib/cmake/zlib, check path
zlib_config_path="./build_thirdparty/zlib/lib/cmake/zlib/zlibConfig.cmake"
install_base_zlib="$install_base/zlib"
if [[ -f "$zlib_config_path" ]]; then
  echo "Zlib found, skipping installation"
else
  echo "Zlib not found, installing"
    mkdir -p "$install_base_zlib"
    cmake -S ./dependencies/zlib -B ./dependencies/zlib/build -DCMAKE_INSTALL_PREFIX="$install_base_zlib" -DCMAKE_BUILD_TYPE=Release
    cmake --build ./dependencies/zlib/build --config Release
    cmake --install ./dependencies/zlib/build
fi

# copy files 
mkdir -p "$current_dir/bin"
cp -r "$install_base_tbb/bin/"* "$current_dir/bin/"
cp -r "$install_base_blosc/bin/"* "$current_dir/bin/"
cp -r "$install_base_zlib/bin/"* "$current_dir/bin/"