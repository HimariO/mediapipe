// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_simple_calculator.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

namespace mediapipe {

// Converts RGB images into luminance images, still stored in RGB format.
// See GlSimpleCalculatorBase for inputs, outputs and input side packets.
class LdrToHdrCalulator : public GlSimpleCalculator {
 public:
  absl::Status Process(CalculatorContext* cc) override;
  // absl::Status Open(CalculatorContext* cc) override;
  static absl::Status GetContract(CalculatorContract* cc);
  
  GpuBufferFormat GetOutputFormat() override;
  absl::Status GlSetup() override;
  absl::Status GlRender(const GlTexture& src, const GlTexture& dst) override;
  absl::Status GlTeardown() override;

 private:
  GLuint program_ = 0;
  GLint frame_;
  GLint src_exp_;
  GLint dst_exp_;
  float packet_src_exp;
  float packet_dst_exp;
};

REGISTER_CALCULATOR(LdrToHdrCalulator);

GpuBufferFormat LdrToHdrCalulator::GetOutputFormat() { return GpuBufferFormat::kRGBAHalf64; }

absl::Status LdrToHdrCalulator::Process(CalculatorContext* cc) {
  return RunInGlContext([this, cc]() -> absl::Status {
    const auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0).Get<GpuBuffer>();
    const auto& src_exp = TagOrIndex(cc->Inputs(), "FRAME_EXPOSURE", 1).Get<float>();
    // LOG(ERROR) << "exp packet: " << src_exp << ", " << dst_exp;

    packet_src_exp = src_exp;
    // packet_dst_exp = dst_exp;

    if (!initialized_) {
      MP_RETURN_IF_ERROR(GlSetup());
      initialized_ = true;
    }

    auto src = helper_.CreateSourceTexture(input);
    int dst_width;
    int dst_height;
    GetOutputDimensions(src.width(), src.height(), &dst_width, &dst_height);
    auto dst = helper_.CreateDestinationTexture(dst_width, dst_height,
                                                GetOutputFormat());

    helper_.BindFramebuffer(dst);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src.target(), src.name());

    MP_RETURN_IF_ERROR(GlBind());
    // Run core program.
    MP_RETURN_IF_ERROR(GlRender(src, dst));

    glBindTexture(src.target(), 0);

    glFlush();

    auto output = dst.GetFrame<GpuBuffer>();

    src.Release();
    dst.Release();

    TagOrIndex(&cc->Outputs(), "VIDEO", 0)
        .Add(output.release(), cc->InputTimestamp());
    TagOrIndex(&cc->Outputs(), "FRAME_EXPOSURE", 1)
        .AddPacket(MakePacket<float>(src_exp).At(cc->InputTimestamp()));

    return absl::OkStatus();
  });
}

absl::Status LdrToHdrCalulator::GetContract(CalculatorContract* cc) {
  TagOrIndex(&cc->Inputs(), "VIDEO", 0).Set<GpuBuffer>();
  TagOrIndex(&cc->Inputs(), "FRAME_EXPOSURE", 1).Set<float>();
  TagOrIndex(&cc->Outputs(), "VIDEO", 0).Set<GpuBuffer>();
  TagOrIndex(&cc->Outputs(), "FRAME_EXPOSURE", 1).Set<float>();
  // Currently we pass GL context information and other stuff as external
  // inputs, which are handled by the helper.
  return GlCalculatorHelper::UpdateContract(cc);
}

absl::Status LdrToHdrCalulator::GlSetup() {
  // Load vertex and fragment shaders
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  const GLchar* frag_src = GLES_VERSION_COMPAT
      R"(
#if __VERSION__ < 130
  #define in varying
#endif  // __VERSION__ < 130

#ifdef GL_ES
  #define fragColor gl_FragColor
  precision highp float;
#else
  #define lowp
  #define mediump
  #define highp
  #define texture2D texture
  out vec4 fragColor;
#endif  // defined(GL_ES)

  in vec2 sample_coordinate;
  uniform sampler2D video_frame;
  uniform highp float src_exp;
  float gamma = 2.2;

  void main() {
    vec4 color = texture2D(video_frame, sample_coordinate);
    fragColor.rgb = pow(color.rgb, gamma) / src_exp;
    fragColor.a = 1.0;
  }

  )";

  // shader program
  GlhCreateProgram(kBasicVertexShader, frag_src, NUM_ATTRIBUTES,
                   (const GLchar**)&attr_name[0], attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  frame_ = glGetUniformLocation(program_, "video_frame");
  src_exp_ = glGetUniformLocation(program_, "src_exp");
  return absl::OkStatus();
}

absl::Status LdrToHdrCalulator::GlRender(const GlTexture& src,
                                           const GlTexture& dst) {
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);
  glUniform1i(frame_, 1);
  glUniform1f(src_exp_, packet_src_exp);  // added 

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);

  return absl::OkStatus();
}

absl::Status LdrToHdrCalulator::GlTeardown() {
  if (program_) {
    glDeleteProgram(program_);
    program_ = 0;
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
