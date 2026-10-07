cmake_minimum_required(VERSION 4.3)
project(HowSpeak LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# SOURCE_SUBDIR points to a missing directory so that only the sources are fetched.
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    SOURCE_SUBDIR none)
FetchContent_Declare(cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.5.0.tar.gz
    URL_HASH SHA256=17e02b4cddc2fa348e5dbdbb582c59a3486fa2b2433e70a0c3bacb871334fd55
    SOURCE_SUBDIR none)
FetchContent_Declare(udpipe
    URL https://github.com/ufal/udpipe/archive/refs/tags/v1.4.0.tar.gz
    URL_HASH SHA256=b6c4c0de45b7f50688a823dc9189f712e4c6a4bb81c61085be513e4da54e9fb9
    SOURCE_SUBDIR none)
FetchContent_Declare(hunspell
    URL https://github.com/hunspell/hunspell/releases/download/v1.7.2/hunspell-1.7.2.tar.gz
    URL_HASH SHA256=11ddfa39afe28c28539fe65fc4f1592d410c1e9b6dd7d8a91ca25d85e9ec65b8
    SOURCE_SUBDIR none)
FetchContent_Declare(imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.9b.tar.gz
    URL_HASH SHA256=21d8a0a565e85dce943e375db00812c2f3f0ab21f3f0f7964e364a63422d7f99
    SOURCE_SUBDIR none)
FetchContent_Declare(glfw
    URL https://github.com/glfw/glfw/releases/download/3.5.1/glfw-3.5.1.zip
    URL_HASH SHA256=ea79bc5feffc254c87291980c2d0bce9acebb68c4983b79f961dcd2cb8a611a0)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen cli11 udpipe hunspell imgui glfw)

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
target_compile_definitions(eigen INTERFACE EIGEN_MPL2_ONLY)

add_library(cli11 INTERFACE)
target_include_directories(cli11 SYSTEM INTERFACE ${cli11_SOURCE_DIR}/include)

file(GLOB_RECURSE UDPIPE_SOURCES CONFIGURE_DEPENDS ${udpipe_SOURCE_DIR}/src/*.cpp)
list(FILTER UDPIPE_SOURCES EXCLUDE REGEX "/src/(udpipe\\.cpp|rest_server/|utils/win_wmain_utf8\\.cpp|morphodita/derivator/derivator_dictionary_encoder\\.cpp)")
add_library(udpipe STATIC ${UDPIPE_SOURCES})
target_include_directories(udpipe PRIVATE ${udpipe_SOURCE_DIR}/src)
target_include_directories(udpipe SYSTEM INTERFACE ${udpipe_SOURCE_DIR}/src_lib_only)
set_target_properties(udpipe PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
target_compile_options(udpipe PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-w> $<$<CXX_COMPILER_ID:MSVC>:/w>)

file(GLOB HUNSPELL_SOURCES ${hunspell_SOURCE_DIR}/src/hunspell/*.cxx)
add_library(hunspell STATIC ${HUNSPELL_SOURCES})
target_compile_definitions(hunspell PUBLIC HUNSPELL_STATIC)
target_include_directories(hunspell SYSTEM INTERFACE ${hunspell_SOURCE_DIR}/src)
set_target_properties(hunspell PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
target_compile_options(hunspell PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-w> $<$<CXX_COMPILER_ID:MSVC>:/w>)

add_library(howspeak_core STATIC
    src/core/Csv.cpp
    src/core/Utf8.cpp
    src/core/Binary.cpp
    src/wals/Database.cpp
    src/nn/Layers.cpp
    src/nn/Imputer.cpp
    src/syntax/Sentence.cpp
    src/syntax/Conllu.cpp
    src/parse/Parser.cpp
    src/parse/CompoundSplitter.cpp
    src/transfer/Grammar.cpp
    src/transfer/Groups.cpp
    src/transfer/Restructure.cpp
    src/transfer/Rules.cpp
    src/transfer/Synthetic.cpp
    src/transfer/Encoder.cpp
    src/transfer/Reorderer.cpp
    src/transfer/Realizer.cpp
    src/transfer/Pipeline.cpp)
target_include_directories(howspeak_core PUBLIC src)
target_link_libraries(howspeak_core PUBLIC eigen udpipe hunspell)
target_compile_definitions(howspeak_core PUBLIC HOWSPEAK_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(HOWSPEAK_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual -Werror)
elseif(MSVC)
    set(HOWSPEAK_WARNINGS /W4 /permissive- /WX /utf-8)
endif()
target_compile_options(howspeak_core PRIVATE ${HOWSPEAK_WARNINGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(howspeak_core PUBLIC $<$<CONFIG:Release>:-O3 -march=native>)
endif()

add_executable(HowSpeak src/app/main.cpp)
target_link_libraries(HowSpeak PRIVATE howspeak_core cli11)
target_compile_options(HowSpeak PRIVATE ${HOWSPEAK_WARNINGS})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(HowSpeak PRIVATE stdc++exp)
endif()
if(MINGW)
    target_link_options(HowSpeak PRIVATE -static-libgcc -static-libstdc++)
endif()

find_package(OpenGL REQUIRED)
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${imgui_SOURCE_DIR}/misc/cpp)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
target_compile_options(imgui PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-w> $<$<CXX_COMPILER_ID:MSVC>:/w>)

add_executable(HowSpeakUI WIN32 src/ui/main.cpp)
target_link_libraries(HowSpeakUI PRIVATE howspeak_core imgui)
target_compile_options(HowSpeakUI PRIVATE ${HOWSPEAK_WARNINGS})
if(MINGW)
    target_link_options(HowSpeakUI PRIVATE -static-libgcc -static-libstdc++)
endif()
