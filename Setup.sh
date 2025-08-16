#!/bin/bash

# =============================================================================
# Ifrit Project Setup Script
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Argument Parsing
# =============================================================================

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Ifrit Project Setup Script

Options:
    --clang-root PATH       Path to Clang installation root directory
    --help                  Show this help message

Examples:
    $0
    $0 --clang-root /usr/lib/llvm-14
    $0 --clang-root "C:/Program Files/LLVM"

EOF
}

# Default values
clang_root=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clang-root)
            clang_root="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate clang_root if provided
if [[ -n "$clang_root" ]]; then
    if [[ ! -d "$clang_root" ]]; then
        log_error "Clang root directory does not exist: $clang_root"
        exit 1
    fi
    log_info "Using Clang root: $clang_root"
else
    log_info "No Clang root specified, using system default"
fi

# =============================================================================
# Environment Setup
# =============================================================================

current_dir=$(pwd)
log_info "Current directory: $current_dir"

# Create base directories
mkdir -p "$current_dir/build_thirdparty/"
mkdir -p "$current_dir/bin/"

# Configuration variables
clang_include_dir="$current_dir/dependencies/"
install_base="$current_dir/build_thirdparty"

# =============================================================================
# Function: Build component with CMake
# =============================================================================
build_component() {
    local name=$1
    local source_dir=$2
    local build_dir=$3
    local install_dir=$4
    local cmake_args=$5
    
    log_info "Building $name..."
    mkdir -p "$install_dir"
    cmake -S "$source_dir" -B "$build_dir" \
          -DCMAKE_INSTALL_PREFIX="$install_dir" \
          -DCMAKE_BUILD_TYPE=Release \
          $cmake_args
    cmake --build "$build_dir" --config Release
    cmake --install "$build_dir"
    log_success "$name build completed"
}

# =============================================================================
# Component: ifrit.reflparser
# =============================================================================
log_info "Setting up ifrit.reflparser..."

ifrit_refl_include_dir="$current_dir/include/"
ifrit_refl_parser_path="./bin/ifrit.reflparser.exe"
target_bin_dir="$current_dir/bin/"

if [[ -f "$ifrit_refl_parser_path" ]]; then
    log_success "ifrit.reflparser found, skipping installation"
else
    log_info "ifrit.reflparser not found, installing..."
    
    cmake_args="-DCUSTOM_CLANG_ROOT=\"$clang_root\" \
                -DCLANG_INCLUDE_DIR=\"$clang_include_dir\" \
                -DADDITIONAL_INCLUDE_DIRS=\"$ifrit_refl_include_dir\" \
                -DCUSTOM_BIN_DIR=\"$target_bin_dir\""
    
    cmake -S ./modules/reflparser \
          -B ./build_thirdparty/ifrit.reflparser \
          -DCMAKE_INSTALL_PREFIX="$target_bin_dir" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCUSTOM_CLANG_ROOT="$clang_root" \
          -DCLANG_INCLUDE_DIR="$clang_include_dir" \
          -DADDITIONAL_INCLUDE_DIRS="$ifrit_refl_include_dir" \
          -DCUSTOM_BIN_DIR="$target_bin_dir"
    
    cmake --build ./build_thirdparty/ifrit.reflparser --config Release
    log_success "ifrit.reflparser installation completed"
fi

# =============================================================================
# Component: TBB (Threading Building Blocks)
# =============================================================================
log_info "Setting up TBB..."

tbb_config_path="./build_thirdparty/tbb/lib/cmake/TBB/TBBConfig.cmake"
install_base_tbb="$install_base/tbb"

if [[ -f "$tbb_config_path" ]]; then
    log_success "TBB found, skipping installation"
else
    log_info "TBB not found, installing..."
    build_component "TBB" \
                   "./dependencies/tbb" \
                   "./dependencies/tbb/build" \
                   "$install_base_tbb" \
                   "-DTBB_TEST=OFF"
fi

# =============================================================================
# Component: Blosc (Compression Library)
# =============================================================================
log_info "Setting up Blosc..."

blosc_config_path="./build_thirdparty/blosc/lib/libblosc.lib"
install_base_blosc="$install_base/blosc"

if [[ -f "$blosc_config_path" ]]; then
    log_success "Blosc found, skipping installation"
else
    log_info "Blosc not found, installing..."
    build_component "Blosc" \
                   "./dependencies/blosc" \
                   "./dependencies/blosc/build" \
                   "$install_base_blosc" \
                   ""
fi

# =============================================================================
# Component: Zlib (Compression Library)
# =============================================================================
log_info "Setting up Zlib..."

zlib_config_path="./build_thirdparty/zlib/lib/cmake/zlib/zlibConfig.cmake"
install_base_zlib="$install_base/zlib"

if [[ -f "$zlib_config_path" ]]; then
    log_success "Zlib found, skipping installation"
else
    log_info "Zlib not found, installing..."
    build_component "Zlib" \
                   "./dependencies/zlib" \
                   "./dependencies/zlib/build" \
                   "$install_base_zlib" \
                   ""
fi

# =============================================================================
# Copy Runtime Dependencies
# =============================================================================
log_info "Copying runtime dependencies to bin directory..."

copy_if_exists() {
    local source=$1
    local dest=$2
    if [[ -d "$source" ]]; then
        cp -r "$source"* "$dest/"
        log_success "Copied $(basename "$source") binaries"
    else
        log_warning "Source directory not found: $source"
    fi
}

copy_if_exists "$install_base_tbb/bin/" "$current_dir/bin/"
copy_if_exists "$install_base_blosc/bin/" "$current_dir/bin/"
copy_if_exists "$install_base_zlib/bin/" "$current_dir/bin/"

# =============================================================================
# Main Project Build
# =============================================================================
log_info "Starting main project build..."

cmake -S ./ -B ./build
cmake --build ./build --config RelWithDebInfo

log_success "Setup completed successfully!"
log_info "Project built with RelWithDebInfo configuration"