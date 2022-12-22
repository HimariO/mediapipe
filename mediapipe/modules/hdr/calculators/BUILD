
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "adjust_exposure_calculator_proto",
    srcs = ["adjust_exposure_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "adjust_exposure_calculator",
    srcs = ["adjust_exposure_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":adjust_exposure_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ldr_to_hdr_calculator",
    srcs = ["ldr_to_hdr_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        # ":ldr_to_hdr_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "concat_image_calculator",
    srcs = ["concat_image_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        # ":ldr_to_hdr_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)
