#pragma once

#include <filesystem>

namespace howspeak::core {

[[nodiscard]] std::filesystem::path executablePath();

// Directory holding data/ and models/: $HOWSPEAK_ROOT, then next to the executable, then the
// Resources folder of a macOS bundle, then the source tree the binary was built from.
[[nodiscard]] std::filesystem::path projectRoot();

}
