#!/bin/sh
# Builds build-macos/HowSpeak.app, a universal (arm64 + x86_64) bundle carrying data/ and models/.
# Needs Xcode 16+ command line tools (xcode-select --install) and CMake 4.3+ (brew install cmake).
set -eu
cd "$(dirname "$0")/.."

for model in models/imputer.bin models/reorderer.bin; do
    [ -f "$model" ] || { echo "missing $model: copy it from the training machine" >&2; exit 1; }
done

cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release
cmake --build build-macos --parallel --target HowSpeakUI HowSpeak
codesign --force --deep --sign - build-macos/HowSpeak.app
lipo -info build-macos/HowSpeak.app/Contents/MacOS/HowSpeak
echo "open build-macos/HowSpeak.app"
