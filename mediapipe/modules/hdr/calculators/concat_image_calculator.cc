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
#include <vector>

#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_simple_calculator.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

namespace mediapipe {

// Converts RGB images into luminance images, still stored in RGB format.
// See GlSimpleCalculatorBase for inputs, outputs and input side packets.
class ConcatImageCalulator : public GlSimpleCalculator {
 public:
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Open(CalculatorContext* cc) override;
  static absl::Status GetContract(CalculatorContract* cc);
  void GetOutputDimensions(int src_width, int src_height,
                            int* dst_width, int* dst_height) override;
  
  // GpuBufferFormat GetOutputFormat() override;
  absl::Status GlSetup() override;
  absl::Status GlRender(const GlTexture& src, const GlTexture& dst) override;
  absl::Status GlTeardown() override;

 private:
  GLuint program_ = 0;
  std::vector<GLint> frames_;
  GLfloat height_;
  GLfloat width_;
  GLfloat num_texture_;

  int src_width;
  int src_height;
  int concat_frames = 3;
  std::vector<GpuBuffer> cache;
};

REGISTER_CALCULATOR(ConcatImageCalulator);


void ConcatImageCalulator::GetOutputDimensions(int src_width, int src_height,
                                                    int* dst_width, int* dst_height) {
    *dst_width = src_width;
    *dst_height = src_height * concat_frames;
}

absl::Status ConcatImageCalulator::Open(CalculatorContext* cc) {
  frames_.resize(concat_frames);
  cc->SetOffset(mediapipe::TimestampDiff(0));
  // options_ = cc->Options<AdjustExposureCalculatorOptions>();

  // Let the helper access the GL context information.
  return helper_.Open(cc);
}

absl::Status ConcatImageCalulator::Process(CalculatorContext* cc) {
  return RunInGlContext([this, cc]() -> absl::Status {
    const auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0).Get<GpuBuffer>();
    cache.push_back(input);
    
    // not enough of frame to create a output
    LOG(ERROR) << "bufferd frame: " << cache.size() << "/ " << concat_frames;

    if (!initialized_) {
      MP_RETURN_IF_ERROR(GlSetup());
      initialized_ = true;
    }

    std::vector<GlTexture> src_textures;
    for (size_t i = 0; i < concat_frames && i < cache.size(); i++)
      src_textures.push_back(helper_.CreateSourceTexture(cache[i]));
    auto src = src_textures[0];
    
    
    int dst_width;
    int dst_height;
    GetOutputDimensions(src.width(), src.height(), &dst_width, &dst_height);
    auto dst = helper_.CreateDestinationTexture(dst_width, dst_height,
                                                GetOutputFormat());
    src_width = src.width();
    src_height = src.height();

    helper_.BindFramebuffer(dst);
    for (size_t i = 0; i < concat_frames && i < cache.size(); i++) {
      auto srctxt = src_textures[i];
      glActiveTexture(GL_TEXTURE1 + i);
      glBindTexture(srctxt.target(), srctxt.name());
    }

    MP_RETURN_IF_ERROR(GlBind());
    // Run core program.
    MP_RETURN_IF_ERROR(GlRender(src, dst));

    // glBindTexture(src.target(), 0);
    for (size_t i = 0; i < concat_frames && i < cache.size(); i++) {
      auto srctxt = src_textures[i];
      glBindTexture(srctxt.target(), 0);
    }

    glFlush();

    auto output = dst.GetFrame<GpuBuffer>();

    for (auto src_txt: src_textures) {
      src_txt.Release();
    }
    dst.Release();

    TagOrIndex(&cc->Outputs(), "VIDEO", 0)
        .Add(output.release(), cc->InputTimestamp());
    // TagOrIndex(&cc->Outputs(), "FRAME_EXPOSURE", 1)
    //     .AddPacket(MakePacket<float>(src_exp).At(cc->InputTimestamp()));
    if (cache.size() >= concat_frames)
      cache.erase(cache.begin());
    return absl::OkStatus();
  });
}

absl::Status ConcatImageCalulator::GetContract(CalculatorContract* cc) {
  TagOrIndex(&cc->Inputs(), "VIDEO", 0).Set<GpuBuffer>();
  // TagOrIndex(&cc->Inputs(), "FRAME_EXPOSURE", 1).Set<float>();
  TagOrIndex(&cc->Outputs(), "VIDEO", 0).Set<GpuBuffer>();
  // TagOrIndex(&cc->Outputs(), "FRAME_EXPOSURE", 1).Set<float>();
  // Currently we pass GL context information and other stuff as external
  // inputs, which are handled by the helper.
  return GlCalculatorHelper::UpdateContract(cc);
}

absl::Status ConcatImageCalulator::GlSetup() {
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
  
  uniform sampler2D video_frame_1;
  uniform sampler2D video_frame_2;
  uniform sampler2D video_frame_3;
  uniform sampler2D video_frame_4;
  
  uniform float height;
  uniform float width;
  uniform float num_texture;

  void main() {
    vec4 color = vec4(0.0, sample_coordinate.x, sample_coordinate.y, 0.0);
    float scale_back = (1.0/ height);
    // color = texture2D(video_frame_3, sample_coordinate);

    if (sample_coordinate.y / height < num_texture) {
      if (sample_coordinate.y / height < 1.0) {
        vec2 pos = vec2(sample_coordinate.x, sample_coordinate.y * scale_back );
        color = texture2D(video_frame_1, pos);
      }
      else if (sample_coordinate.y / height < 2.0) {
        vec2 pos = vec2(sample_coordinate.x, (sample_coordinate.y - height * 1.0) * scale_back );
        color = texture2D(video_frame_2, pos);
      }
      else if (sample_coordinate.y / height < 3.0) {
        vec2 pos = vec2(sample_coordinate.x, (sample_coordinate.y - height * 2.0) * scale_back );
        color = texture2D(video_frame_3, pos);
      }
      else if (sample_coordinate.y / height < 4.0) {
        vec2 pos = vec2(sample_coordinate.x, (sample_coordinate.y - height * 3.0) * scale_back );
        color = texture2D(video_frame_4, pos);
      }
    }
    fragColor.rgb = color.rgb;
    fragColor.a = 1.0;
  }

  )";

  // shader program
  GlhCreateProgram(kBasicVertexShader, frag_src, NUM_ATTRIBUTES,
                   (const GLchar**)&attr_name[0], attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";

  for (int i = 0; i < concat_frames; i++){
    const char* uni_name = ("video_frame_" + std::to_string(i + 1)).c_str();
    frames_[i] = glGetUniformLocation(program_, uni_name);
    RET_CHECK(frames_[i]) << "GL initializing " << i << "\'th texture";
  }
  height_ = glGetUniformLocation(program_, "height");
  width_ = glGetUniformLocation(program_, "width");
  num_texture_ = glGetUniformLocation(program_, "num_texture");
  return absl::OkStatus();
}

absl::Status ConcatImageCalulator::GlRender(const GlTexture& src,
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
  for (size_t i = 0; i < concat_frames; i++)
    glUniform1i(frames_[i], i + 1);
  glUniform1f(height_, 1.0 / static_cast<float>(concat_frames));  // added 
  glUniform1f(width_, 1.0 / static_cast<float>(concat_frames));  // added 
  glUniform1f(num_texture_, cache.size());  // added 

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

absl::Status ConcatImageCalulator::GlTeardown() {
  if (program_) {
    glDeleteProgram(program_);
    program_ = 0;
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
