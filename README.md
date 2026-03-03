# Waterfall-SDR

This program provides a Waterfall display for frequency analysis of radio waves. It uses the `librtlsdr` to 
interface with and collect samples from the RTL-SDR Blog Model V3 dongle. It feeds these IQ samples to a 
custome digital signal processing block (DSP), which uses Hann Window and In-Place Fast Fourier techniques
to unmix signals. The result is a frequency-domain view from complex samples in the time domain. 

For the GUI displaying the waterfall, SDL is used. 

You must have **RTL-SDR Blog Model V3 dongle**

## Build

### Dependencies

- `librtlsdr `
- `libusb-1.0`
- SDL (included as a git submodule). vendored/SDL, built automatically by CMake
- Ubuntu / Debian / WSL recommended

#### 1. Clone the repo with submodules
```
git clone --recurse-submodules https://github.com/kaaldana1/waterfall-sdr
cd waterfall-sdr
```

#### 2. Install system packages
```
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
  librtlsdr-dev libusb-1.0-0-dev
```

#### 3. Configure + build (CMake)

From repo root:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The binary will be:
```
./build/waterfall
```


