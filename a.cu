#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <memory>

#include "config.cuh"
#include "mathlib.cuh"

enum RenderActionType {
  RenderActionNop = 0,
  RenderActionWrite = 1,
  RenderActionAdd = 2,
  RenderActionAlphaBlend = 3
};

class FractalObject3D {
public:
  // CUDA_FN vec3 iterFormula(const vec3 &v) const {return new_vec3(0.f);}
  CUDA_FN vec3 iterFormula(const vec3 &v) const;
  // CUDA_FN float estimateDistance(const vec3 &v) const {return 0.f;}
  // CUDA_FN bool hasEscaped(const vec3 &v) const {return false;}
  // CUDA_FN bool hasEscaped(const vec3 &v) const;
  // CUDA_FN int escapeTime(const vec3 &v0, int max_iter) const {
    // vec3 v = v0; int i=0;
    // for(; i<max_iter; ++i) {
      // if (hasEscaped(v))
        // break;
      // v = iterFormula(v);
    // }
    // return i;
  // }
};

class MengerSpongeFractal {
public:
  CUDA_FN vec3 iterFormula(const vec3 &v, const vec3 &v0) const {
    auto fn = [](float x) {
      if (fabsf(x)<(1.f/3)) { return x*3.f;
      } else { return x*3.f + ((x>0.f) ? -2.f : 2.f); }
    };
    return new_vec3(fn(v.x), fn(v.y), fn(v.z));
  }
  CUDA_FN bool hasEscaped(const vec3 &v) const {
    auto fn = [](float x) {
      float y = fabsf(x);
      if (y>1.f) return 2.f;
      return fabsf(y - (2.f/3)) > (1.f/3) ? 1.f : 0.f;
    };
    return (fn(v.x)+fn(v.y)+fn(v.z)) > 1.f;
  }
  CUDA_FN int escapeTime(const vec3 &v0, int max_iter) const {
    vec3 v = v0; int i=0;
    for(; i<max_iter; ++i) {
      if (hasEscaped(v))
        break;
      v = iterFormula(v, v0);
    }
    return i;
  }
};

class CubeSquareFractal {
public:
  CUDA_FN vec3 iterFormula(const vec3 &v, const vec3 &v0) const {
    float largest = v.x; unsigned idx=0;
    if (fabsf(largest) < fabsf(v.y)) {
      largest = v.y; idx=1;
    }
    if (fabsf(largest) < fabsf(v.z)) {
      largest = v.z; idx=2;
    }
    float sign = largest > 0.f ?1.f :-1.f;
    // float sign = 1.f;
    vec3 ret;
    switch(idx) {
    case 0: ret = new_vec3(
        v.x*v.x - v.y*v.y - v.z*v.z, 2*v.x*v.y, 2*v.x*v.z
    ); break;
    case 1: ret = new_vec3(
        2*v.y*v.x, v.y*v.y - v.x*v.x - v.z*v.z, 2*v.y*v.z
    ); break;
    case 2: ret = new_vec3(
        2*v.z*v.x, 2*v.z*v.y, v.z*v.z - v.x*v.x - v.y*v.y
    ); break;
    }
    return (ret*sign) + v0*0.05;
  }
  CUDA_FN vec3 iterFormula(const vec3 &v, const vec3 &v0, mat3x3 &J) const {
    // this also computes jacobian matrix
    float largest = v.x; unsigned idx=0;
    if (fabsf(largest) < fabsf(v.y)) {
      largest = v.y; idx=1;
    }
    if (fabsf(largest) < fabsf(v.z)) {
      largest = v.z; idx=2;
    }
    float sign = largest > 0.f ? 1.f :-1.f;
    // float sign = 1.f;
    vec3 ret;
    switch(idx) {
    case 0: ret = new_vec3(
        v.x*v.x - v.y*v.y - v.z*v.z, 2*v.x*v.y, 2*v.x*v.z
    );
    J = new_mat3x3(
        v.x,-v.y,-v.z,
        v.y, v.x, 0.f,
        v.z, 0.f, v.x
    ) * sign; break;
    case 1: ret = new_vec3(
        2*v.y*v.x, v.y*v.y - v.x*v.x - v.z*v.z, 2*v.y*v.z
    );
    J = new_mat3x3(
        v.y, v.x, 0.f,
       -v.x, v.y,-v.z,
        0.f, v.z, v.y
    ) * sign; break;
    case 2: ret = new_vec3(
        2*v.z*v.x, 2*v.z*v.y, v.z*v.z - v.x*v.x - v.y*v.y
    );
    J = new_mat3x3(
        v.z, 0.f, v.x,
        0.f, v.z, v.y,
       -v.x,-v.y, v.z
    ) * sign; break;
    }
    return (ret*sign) + v0*0.05;
  }
  CUDA_FN bool hasEscaped(const vec3 &v, float norm2=4.f) const {
    return dot(v,v) > norm2;
  }
  CUDA_FN int escapeTime(const vec3 &v0, int max_iter) const {
    // cheapest inside/outside estimator
    vec3 v = v0; int i=0;
    for(; i<max_iter; ++i) {
      if (hasEscaped(v))
        break;
      v = iterFormula(v, v0);
    }
    return i;
  }
  CUDA_FN float estimateDistance(const vec3 &v0, int max_iter, int count_offset=0, float stop_norm2=1e3f) const {
    // slightly more expensive than escapeTime
    vec3 v = new_vec3(v0.x,v0.y,v0.z);
    float norm2, ret=0.f; int i=0;
    for(; i<max_iter; ++i) {
      norm2 = dot(v,v);
      if (norm2 > stop_norm2) { break; }
      v = iterFormula(v, v0);
    }
    ret = powf(norm2, exp2f(-i-1-count_offset)) - 1.f;
    return ret;
  }
  CUDA_FN vec3 estimateNormal(const vec3 &v0, int max_iter, float stop_norm2=1e6f) const {
    // only use this for shading purposes
    // this is more expensive than escape_time || distance_est
    // only works with points outside fractal
    vec3 v = new_vec3(v0.x,v0.y,v0.z); int i=0;
    mat3x3 J, J_accum = new_mat3x3_ident();
    float norm2;
    for(; i<max_iter; ++i) {
      norm2 = dot(v,v);
      if (norm2 > stop_norm2) {
        break;
      }
      v = iterFormula(v, v0, J);
      J_accum = dot(J, J_accum);
      // J_accum = dot(J_accum, J);
      // normalize every 4 iterations
      if (3 == (i&3)) J_accum = J_accum.normalized_l0();
    }
    // gotta use safe normalization here
    return dot(J_accum.adjugate(), v).normalized();
    // return dot(J_accum, v).normalized();
    // return dot(J_accum.adjugate().T(), v).normalized();
    // return v.normalized();
  }
};

struct ColorGradientFunc {
  CUDA_FN vec4 operator() (float x);
};
struct BiColorGradientFunc {
  vec4 color0, color1;
  BiColorGradientFunc(const vec4 &c0, const vec4 &c1): color0(c0),color1(c1) {}
  CUDA_FN vec4 operator() (float x) {
    return color1*x + color0*(1.f-x);
  }
};
struct TriColorGradientFunc {
  vec4 color0, color1, color2;
  TriColorGradientFunc(
      const vec4 &c0, const vec4 &c1, const vec4 &c2): color0(c0),color1(c1),color2(c2) {}
  CUDA_FN vec4 operator() (float x) {
    if (x<0.5f) {
      x = x*2.f;
      return color0*(1.f-x) + color1*x;
    } else {
      x = x*2.f-1.f;
      return color1*(1.f-x) + color2*x;
    }
  }
};
struct QuadColorGradientFunc {
  vec4 color0, color1, color2, color3;
  QuadColorGradientFunc(
    const vec4 &c0, const vec4 &c1,
    const vec4 &c2, const vec4 &c3
  ): color0(c0),color1(c1),color2(c2),color3(c3) {}
  CUDA_FN vec4 operator() (float x) {
    if (x<(1.f/3)) {
      x = x*3.f;
      return color0*(1.f-x) + color1*x;
    } else if (x<(2.f/3)) {
      x = x*3.f-1.f;
      return color1*(1.f-x) + color2*x;
    } else {
      x = x*3.f-2.f;
      return color2*(1.f-x) + color3*x;
    }
  }
};

struct Camera {
  vec3 screenToWorld(const vec3 &point) const;
  vec3 worldToScreen(const vec3 &point) const;
};

struct ProjectiveCamera : public Camera {
  // camera based on linear fractional transform
  mat4x4 c2w_mat, w2c_mat;
  ProjectiveCamera(): c2w_mat(new_mat4x4_ident()),w2c_mat(new_mat4x4_ident()) {}
  ProjectiveCamera(
      const mat4x4 &c2w_mat_);
  void setupPersp(const vec3 &position, const vec3 &right, const vec3 &up, float z_near, float z_far) {
    auto &self = *this;
    vec3 fwd_unit = cross(up, right).normalized_unsafe();
    auto proj_mat = new_mat4x4_projection(z_far, z_far/z_near);
    auto linear_mat = new_mat4x4_linear(right, up, fwd_unit);
    auto translation_mat = new_mat4x4_translation(position + fwd_unit*z_near);
    self.c2w_mat = dot(translation_mat, dot(linear_mat, proj_mat));
    self.w2c_mat = self.c2w_mat.inverse();
  }
  void setupOrtho(const vec3 &position, const vec3 &right, const vec3 &up, float z_near, float z_far) {
    auto &self = *this;
    float right_norm, up_norm;
    vec3 right_unit = right.normalized_unsafe(right_norm);
    vec3 up_unit = up.normalized_unsafe(up_norm); 
    vec3 fwd_unit = cross(right_unit, up_unit);
    self.c2w_mat = new_mat4x4_projection(z_far-z_near, 1.f);
    self.c2w_mat = dot(new_mat4x4_linear(right, up, fwd_unit), self.c2w_mat);
    self.c2w_mat = dot(new_mat4x4_translation(position + fwd_unit*z_near), self.c2w_mat);
    self.w2c_mat = self.c2w_mat.inverse();
  }
  void setupForwardMatrix(const mat4x4 &mat) {
    c2w_mat = mat;
    w2c_mat = mat.inverse();
  }
  void setupBackwardMatrix(const mat4x4 &mat) {
    w2c_mat = mat;
    c2w_mat = mat.inverse();
  }
  void applyWorldTransform(const mat4x4 &mat) {
    auto &self = *this;
    self.c2w_mat = dot(self.c2w_mat, mat);
    self.w2c_mat = dot(self.w2c_mat, mat.inverse());
  }
  void applyCameraTransform(const mat4x4 &mat) {
    auto &self = *this;
    self.w2c_mat = dot(self.w2c_mat, mat);
    self.c2w_mat = dot(self.c2w_mat, mat.inverse());
  }
  vec3 screenToWorld(vec3 &point) {
    return dot(c2w_mat, point);
  }
  vec3 worldToScreen(vec3 &point) {
    return dot(w2c_mat, point);
  }
};

struct Ray {
  // 8x float32 in total
  vec3 position;
  float full_len;
  vec3 unit_direction;
  float cur_len;
  CUDA_FN vec3 at(float x) {
    return this->position + this->unit_direction*x;
  }
  CUDA_FN vec3 at_current() {
    return this->at(this->cur_len);
  }
};
struct IndexedRay {
  unsigned idx;
  Ray ray;
};
struct RayPtr {
  uint32_t idx;
  float depth;
};

struct DirectRayLookup {
  Ray *ray_buffer;
  unsigned size;
  __device__ bool isOutOfBound(unsigned idx) const { return idx >= size;}
  __device__ IndexedRay operator [](unsigned idx) const {
    return {idx, ray_buffer[idx]};
  }
};

struct IndexedRayLookup {
  Ray *ray_buffer;
  RayPtr *depth_buffer;
  unsigned size;
  __device__ bool isOutOfBound(unsigned idx) const { return idx >= size;}
  __device__ IndexedRay operator [](unsigned idx) const {
    auto ray_idx = depth_buffer[idx].idx;
    return {ray_idx, ray_buffer[ray_idx]};
  }
};

struct EnvMap {
  __device__ vec4 lookup(const vec3 &normal);
  __device__ vec4 lookupUnit(const vec3 &normal);
};

// maps vec3 -> color
struct DefaultEnvMap {
  __device__ vec4 lookupUnit(const vec3 &normal) {
    return new_vec4(.5f, .5f, .5f, 1.f) + new_vec4(normal*0.5f, 0.f);
  }
  __device__ vec4 lookup(const vec3 &normal) {
    return lookupUnit(normal.normalized());
  }
};

/*
struct DirectRayView {
  Ray *ray_buffer;
  unsigned size;
  __device__ bool outOfBound(unsigned idx) const {
    return idx >= size;
  }
  __device__ Ray operator [](unsigned idx) const {
    return ray_buffer[idx];
  }
};
struct DepthCacheRayView {
  Ray *ray_buffer;
  unsigned size;
  RayPtr *depth_cache;
  unsigned cache_size;
  __device__ bool outOfBound(unsigned idx) const {
    return idx >= cache_size;
  }
  __device__ Ray operator [](unsigned idx) const {
    RayPtr ray_ptr = depth_cache[idx];
    Ray ray = ray_buffer[ray_ptr.idx];
    ray.cur_len = ray_ptr.depth;
  }
};
*/

namespace unittest {
  std::ostream& operator <<(std::ostream &os, const Ray &ray) {
    return os << "Ray["
      << ray.position << ", " << ray.unit_direction << ", "
      << ray.cur_len << '/' << ray.full_len << ']';
  }
}
struct DeviceRng {
  curand::curandState_t state;
  __device__ DeviceRng(int seed, unsigned idx, int offset=0) {
    curand::curand_init(1337, idx, 0, &(this->state));
  }
  __device__ inline unsigned int next() {
    return curand::curand(&state);
  }
  __device__ inline int nexti() {
    return static_cast<int>(next());
  }
  __device__ float rand() { // [0.f, 1.f]
    return (float)next()*(1.f/4294967295);
  }
  __device__ float rand2() { // [-1.f, 1.f]
    return (float)nexti()*(2.f/4294967295) + (1.f/4294967296);
  }
  // uniform random vector on unit sphere
  __device__ vec3 randUnitVec3() {
    auto fast_ierf = [](float x) {
      float x2 = x*x; // accurate enough for our application
      float nr = -0.0171826f, dr = -0.0694062;
      nr = nr*x2 + 0.364712f; dr = dr*x2 + 0.676325f;
      nr = nr*x2 - 1.15157f;  dr = dr*x2 - 1.56121f;
      nr = nr*x2 + 0.886227f; dr = dr*x2 + 1.f;
      return (nr*x) / dr;
    };
    return new_vec3( // normal distribution has spherical symmetry
      fast_ierf(rand2()), fast_ierf(rand2()), fast_ierf(rand2())
    ).normalized_unsafe();
  }
};

__device__ Ray _initRay(
    unsigned idx_x, unsigned idx_y,
    float half_w, float half_h,
   unsigned thread_idx, const unsigned rng_offset,
    const mat4x4 c2w_mat, const float init_len=0.f, const bool stochastic=false) {
  Ray ray;
  float u,v;
  if (stochastic) {
    DeviceRng rng(cfg::RNG_SEED, thread_idx+rng_offset, rng_offset);
    u = (float)(idx_x-half_w+rng.rand()-0.5f) / half_w;
    v = (float)(half_h-idx_y+rng.rand()-0.5f) / half_w;
  } else {
    u = (float)(idx_x-half_w) / half_w;
    v = (float)(half_h-idx_y) / half_w;
  }
  vec3 position = dot(c2w_mat, new_vec3(u, v, 0.f));
  vec3 direction = dot(c2w_mat, new_vec3(u, v, 1.f)) - position;
  float len;
  direction = direction.normalized_unsafe(len);
  ray.position = position;
  ray.full_len = len;
  ray.unit_direction = direction;
  ray.cur_len = init_len;
  return ray;
}

// initiailize ray buffer to a rectagular buffer
template<bool stochastic=false>
CUDA_KERNEL initRayBuffer_cukern(
    Ray *ray_buffer, const int w, const int h, const int rng_offset, const mat4x4 c2w_mat) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  const auto idx_x = idx%w, idx_y = idx/w;
  const float half_w = ((float)w-1.f)*0.5f, half_h = ((float)h-1.f)*0.5f;
  if (idx_y>=h) return;
  ray_buffer[idx] = _initRay(
      idx_x, idx_y, half_w, half_h,
      idx, rng_offset,
      c2w_mat, 0.f, stochastic);
}

// initialize 
template<bool stochastic=false>
CUDA_KERNEL initRayBufferWithDepthCache_cukern(
    Ray *ray_buffer, const int w, const int h,
    RayPtr *depth_cache, int depth_cache_size,
    const int rng_offset, const mat4x4 c2w_mat) {
  const auto thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (thread_idx > depth_cache_size) return;
  RayPtr ray_ptr = depth_cache[thread_idx];
  const auto idx = ray_ptr.idx;
  const auto idx_x = idx%w, idx_y = idx/w;
  const float half_w = ((float)w-1.f)*0.5f, half_h = ((float)h-1.f)*0.5f;
  ray_buffer[idx] = _initRay(
      idx_x, idx_y, half_w, half_h,
      idx, rng_offset,
      c2w_mat, ray_ptr.depth, stochastic);
}

// this kernel only launches one block
CUDA_KERNEL renderDepthCache_cukern(
    RayPtr *depth_cache, int *depth_cache_size,
    const Ray *ray_buffer, const int ray_buffer_size
    ) {
  const unsigned warp_idx = threadIdx.x / 32;
  const unsigned lane_idx = threadIdx.x % 32;
  int offset = 0;
  __shared__ unsigned warp_sum_shm[32];
  for (unsigned idx = threadIdx.x; idx<ray_buffer_size; idx+=blockDim.x) {
    const Ray &ray = ray_buffer[idx];
    bool predicate = (ray.cur_len < ray.full_len);
    unsigned sum = block_accum_sum(predicate, lane_idx, warp_idx, warp_sum_shm);
    if (predicate) {
      depth_cache[offset+sum] = {idx, ray.cur_len};
    }
    offset += warp_sum_shm[31];
  }
  if (threadIdx.x == 0) depth_cache_size[0] = offset;
}

// kernel for testing purposes
// finds intersection of ray and unit sphere
CUDA_KERNEL rtUnitSphere_cukern(Ray *ray_buffer, int size) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=size) return;
  Ray ray = ray_buffer[idx];
  do {
    float p_norm2 = dot(ray.position, ray.position);
    if (p_norm2 <= 1.f) {
      ray.cur_len = 0.f;
      break;
    }
    float d_dot_p = dot(ray.unit_direction, ray.position);
    if (d_dot_p > 0.f) {
      ray.cur_len = ray.full_len;
      break;
    }
    vec3 u = ray.unit_direction * d_dot_p;
    vec3 v =ray.position - u;
    float diff_norm2 = 1.f - dot(v,v);
    if (diff_norm2 < 0.f) {
      ray.cur_len = ray.full_len;
      break;
    }
    ray.cur_len = fminf(-(d_dot_p + sqrtf(diff_norm2)), ray.full_len);
  } while(0);
  ray_buffer[idx] = ray;
}

template<typename fractal_t>
float CUDA_FN rtFractalBisectSearch(
    fractal_t fractal, int max_iter,
    Ray &ray, const float depth, const float min_dist=1e-6f) {
  // given two points inside/outside fractal
  // find intersection by bisection
  // fractal_t fractal: fractal object
  // int max_iter: max iteration for fractal.estimateDistance
  // Ray &ray: ray object which is currently outside fractal
  // float depth: depth at which ray is inside fractal
  // float min_dist: distance threshould to stop search
  // float max_norm2: max norm^2 for fractal.estimateDistance
  float l=ray.cur_len, r=depth, m;
  int etime;
  do {
    m = 0.5f * (l+r);
    etime = fractal.escapeTime(ray.position + ray.unit_direction*m, max_iter);
    if (etime == max_iter) {
      r = m;
    } else {l = m;}
  } while((r-l)>min_dist);
  return l;
}

// a very simple fixed-step ray-marching kernel
// for debugging purposes
template<typename fractal_t>
CUDA_KERNEL rtSimpleRaymarch_cukern(Ray *ray_buffer, int size, const fractal_t fractal) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=size) return;
  Ray ray = ray_buffer[idx];
  const float total_steps = 1000.f;
  const int max_iter = cfg::MAX_ITER;
  for(float i=0.f; i<1.f; i+=(1.f/total_steps)) {
    float cur_len = ray.full_len*i;
    vec3 v = ray.position + ray.unit_direction * cur_len;
    if (fractal.escapeTime(v, max_iter) == max_iter)
      break;
    ray.cur_len = cur_len;
  }
  ray_buffer[idx] = ray;
}

// ray-marching with distance estimation and bisection search
// for primary ray-cast
template<typename fractal_t, bool stochastic=false>
CUDA_KERNEL rtRaymarch_cukern(
    Ray *ray_buffer, int size, int rng_offset,
    const fractal_t fractal, const int max_iter) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=size) return;
  DeviceRng rng(cfg::RNG_SEED, idx+rng_offset, 0);
  Ray ray = ray_buffer[idx];
  const float max_step = 1e-2f, min_step = 1e-4f;
  const float min_dist = 1e-6f; // distance at which to stop
  float cur_len = ray.cur_len, d, delta_len;
  for(;;) {
    vec3 v = ray.at(cur_len);
    d = fractal.estimateDistance(v, max_iter);
    if (d == 0.f) {
      ray.cur_len = rtFractalBisectSearch(fractal, max_iter, ray, cur_len, min_dist);
      break;
    }
    ray.cur_len = cur_len;
    if (d < min_dist) break;
    if (d > max_step) {
      delta_len = max_step;
    } else if (d<min_step) {
      delta_len = min_step;
    } else {
      delta_len = d;
    }
    if (stochastic) {
      cur_len += delta_len * rng.rand();
    } else {
      cur_len += delta_len;
    }
    if (cur_len > ray.full_len) {ray.cur_len = ray.full_len; break;}
  }
  ray_buffer[idx] = ray;
}

// colorize current depth to framebuffer
template<typename gradient_fn_t>
CUDA_KERNEL shaderDepth_cukern(
    vec4 *framebuffer, Ray *ray_buffer, int size, gradient_fn_t gradient_fn) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=size) return;
  const Ray &ray = ray_buffer[idx];
  const float interp = ray.cur_len / ray.full_len;
  framebuffer[idx] = gradient_fn(interp);
}

// colorize current depth cache to framebuffer
template<typename gradient_fn_t>
CUDA_KERNEL shaderDepthCache_cukern(
    vec4 *framebuffer, int fb_size,
    RayPtr *depth_cache, int cache_size,
    gradient_fn_t gradient_fn, float depth_scale=1.f) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx>=cache_size) return;
  const RayPtr &ray_ptr = depth_cache[idx];
  const float interp = erff(ray_ptr.depth * depth_scale);
  framebuffer[ray_ptr.idx] = gradient_fn(interp);
}

// estimate normal vector, then render to framebuffer
template<RenderActionType render_action, typename fractal_t, typename RayLookup_t, typename EnvMap_t>
CUDA_KERNEL shaderNormal_cukern(
    vec4 *framebuffer,
    RayLookup_t ray_lookup,
    EnvMap_t env_map,
    const fractal_t fractal,
    const int max_iter=cfg::MAX_ITER_FOR_NORMAL) {
  const auto thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (ray_lookup.isOutOfBound(thread_idx)) return;
  IndexedRay idx_ray = ray_lookup[thread_idx];
  Ray &ray = idx_ray.ray;
  const auto idx = idx_ray.idx;
  vec4 ret;
  const float norm2 = 1e12f;
  if (ray.cur_len == ray.full_len) {
    ret = new_vec4(0.f);
  } else {
    vec3 normal = fractal.estimateNormal(ray.at_current(), max_iter, norm2);
    ret = env_map.lookupUnit(normal);
    // ret = new_vec4(normal*0.5f + new_vec3(0.5f, 0.5f, 0.5f), 1.f);
  }
  switch (render_action) {
    case RenderActionNop:
      break;
    case RenderActionWrite:
      framebuffer[idx] = ret;
      break;
    case RenderActionAdd:
      framebuffer[idx] += ret;
      break;
    case RenderActionAlphaBlend:
      // TODO.feat
      break;
    default:
      break;
  }
}

CUDA_KERNEL AXPY_cukern(float *data, int size, float x, float y) {
  const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx>=size) return;
  data[idx] = data[idx]*x + y;
}

template<typename fractal_t>
class RenderDevice {
#ifdef DEBUG_BUILD
  public:
#else
  protected:
#endif
    ProjectiveCamera camera;
    Ray *ray_buffer;
    vec4 *framebuffer;
    RayPtr *depth_cache;
    int depth_cache_size;
    int w, h;
    int rng_offset;
  public:
  RenderDevice(int w, int h):
    w(w),h(h),
    rng_offset(0),
    depth_cache(nullptr),
    depth_cache_size(0)
  {
    auto &self = *this;
    auto n = w*h;
    cudaMallocTyped(&self.ray_buffer, n);
    cudaMallocTyped(&self.framebuffer, n);
    cudaMemset(self.framebuffer, 0, n*sizeof(decltype(self.framebuffer)));
  }
  ~RenderDevice() {
    auto &self = *this;
    cudaFree(self.ray_buffer);
    cudaFree(self.framebuffer);
    self.clearDepthCache();
  }
  void setCamera(const ProjectiveCamera &_camera) {
    auto &self = *this;
    self.camera = _camera;
  }
  void initRayBuffer(bool use_depth_cache=false, const bool stochastic=false) {
    auto &self = *this;
    auto total_pels = use_depth_cache ? self.depth_cache_size : self.w*self.h;
    const unsigned grid_dim = (total_pels+255)/256, block_dim = 256;
    if (!use_depth_cache) {
      initRayBuffer_cukern<false><<<grid_dim, block_dim>>>(
          self.ray_buffer,
          self.w, self.h, self.rng_offset,
          self.camera.c2w_mat );
    } else {
      initRayBufferWithDepthCache_cukern<true><<<grid_dim, block_dim>>>(
          self.ray_buffer, self.w, self.h,
          self.depth_cache, self.depth_cache_size,
          self.rng_offset, self.camera.c2w_mat );
    }
    self.rng_offset += total_pels;
  }
  void renderDepthCache(const int max_iter=cfg::MAX_ITER_FOR_DEPTH_CACHE) {
    // TODO.feat
    auto &self = *this;
    int total_pels = self.w * self.h;
    const int grid_dim = (total_pels+255)/256, block_dim = 256;
    fractal_t fractal;
    printf("ray marching ..."); fflush(stdout);
    rtRaymarch_cukern<fractal_t, true><<<grid_dim, block_dim>>>(
        ray_buffer, total_pels, rng_offset, fractal, max_iter);
    printf(" done\n");

    int *gpu_depth_cache_size;
    cudaMalloc(&gpu_depth_cache_size, sizeof(int));
    printf("building depth cache ..."); fflush(stdout);
    renderDepthCache_cukern<<<1, 1024>>>(
        self.depth_cache, gpu_depth_cache_size,
        self.ray_buffer, total_pels);
    cudaMemcpy(
        &self.depth_cache_size, gpu_depth_cache_size,
        sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_depth_cache_size);
    printf(" done\n");
    auto color_fn = TriColorGradientFunc(
      new_vec4(1.f, 0.f, 0.f, 1.f),
      new_vec4(0.f, 1.f, 0.f, 1.f),
      new_vec4(0.f, 0.f, 1.f, 1.f)
    );
    /*
    shaderDepthCache_cukern<<<grid_dim, block_dim>>>(
        self.framebuffer, self.w*self.h,
        self.depth_cache, self.depth_cache_size,
        color_fn, 0.3f
    );
    shaderDepth_cukern<<<grid_dim, block_dim>>>(
      self.framebuffer, self.ray_buffer, total_pels, color_fn
    );
    */
  }
  void prepareDepthCache() {
    auto &self = *this;
    if (!self.depth_cache) {
      cudaMallocTyped(&self.depth_cache, self.w * self.h);
    }
  }
  void clearDepthCache() {
    auto &self = *this;
    if (self.depth_cache) {
      cudaFree(self.depth_cache);
      self.depth_cache = nullptr;
    }
    self.depth_cache_size = 0;
  }
  void render(bool use_depth_cache=false, const int max_iter=cfg::MAX_ITER) {
    auto &self = *this;
    unsigned total_pels=0;
    if (!use_depth_cache) total_pels = self.w*self.h;
    else total_pels = self.depth_cache_size;
    if (!total_pels) {
      printf("[W] no pixel being rendered, abort\n");
      return;
    };

    const int grid_dim = (total_pels+255)/256, block_dim = 256;
    // rendering
    fractal_t fractal;
    rtRaymarch_cukern<fractal_t, true><<<grid_dim, block_dim>>>(
        self.ray_buffer, total_pels, rng_offset, fractal, max_iter);
    rng_offset += total_pels;
    // rtSimpleRaymarch_cukern<<<grid_dim, block_dim>>>(ray_buffer, total_pels, fractal);
    
    // shading
    auto color_function = QuadColorGradientFunc(
      new_vec4(1.f, 0.f, 0.f, 1.f),
      new_vec4(0.f, 1.f, 1.f, 1.f),
      new_vec4(1.f, 0.f, .5f, 1.f),
      new_vec4(0.f, 0.f, 0.f, 1.f)
    );
    // shaderDepth_cukern<<<grid_dim, block_dim>>>(
        // framebuffer, ray_buffer, total_pels,
        // color_function
    // );
    if (!use_depth_cache) {
      shaderNormal_cukern<RenderActionAdd><<<grid_dim, block_dim>>>(
          self.framebuffer,
          DirectRayLookup{self.ray_buffer, total_pels},
          DefaultEnvMap{},
          fractal);
    } else {
      shaderNormal_cukern<RenderActionAdd><<<grid_dim, block_dim>>>(
          self.framebuffer,
          IndexedRayLookup{self.ray_buffer, self.depth_cache, total_pels},
          DefaultEnvMap{},
          fractal);
    }
    cudaDeviceSynchronize();
  }
  void scaleFramebuffer(float x, float bias=0.f) {
    auto &self = *this;
    int total_pels=self.w*self.h*4;
    AXPY_cukern<<<(total_pels+255)/256, 256>>>((float*)self.framebuffer, total_pels, x, bias);
  }

  void zeroFramebuffer() {
    auto &self = *this;
    if (self.framebuffer) {
      cudaMemset(self.framebuffer, 0,
          self.w*self.h*sizeof(decltype(*self.framebuffer)));
    }
  }
  void exportFramebufferToFile(std::ofstream &out_file) {
    vec4 *row_buffer = new vec4[w];
    uchar4 *row_color_buffer = new uchar4[w];
    for(int i=0; i<w*h; i+=w) {
      cudaMemcpy(row_buffer, framebuffer+i, w*sizeof(decltype(*row_buffer)), cudaMemcpyDeviceToHost);
      for(int j=0; j<w; ++j) {
        row_color_buffer[j] = (uchar4)(row_buffer[j]);
      }
      out_file.write((char*)row_color_buffer, sizeof(uchar4)*w);
    }
    delete [] row_buffer;
    delete [] row_color_buffer;
  }
  void exportFramebufferToFile(const std::string &filename) {
    using std::ios;
    std::ofstream out_file;
    out_file.open(filename, ios::out|ios::binary);
    this->exportFramebufferToFile(out_file);
    out_file.close();
  }
};

#ifdef DEBUG_BUILD
namespace unittest {
void test_mat4x4_inverse() {
  const mat4x4 m = new_mat4x4(
    0.5603606f , 0.24954285f, 0.34497562f, 0.934563f  ,
    0.8690652f , 0.5952718f , 0.4876602f , 0.29138133f,
    0.30592227f, 0.4398568f , 0.57195455f, 0.20199597f,
    0.23505567f, 0.54009676f, 0.98911834f, 0.1063792f
  );
  const mat4x4 m_inv = m.inverse();
  const mat4x4 m_inv_ref = new_mat4x4(
     0.39788654f,  2.6760447f, -7.0933013f,  2.6435506f,
    -1.6476806f , -2.1576693f, 14.195851f , -6.5702577f,
     0.69603646f,  0.6798256f, -6.368396f ,  4.1155777f,
     1.0144756f , -1.2793598f,  2.8133738f, -1.3498874f
  );
  std::cout << m_inv << '\n' << m_inv_ref << std::endl;
}
void test_mat3x3_inverse() {
  const mat3x3 m = new_mat3x3(
    -0.27483684,  0.60266045, -0.47038684,
     0.08806632, -0.06213188, -0.2474315 ,
    -0.0315542 ,  0.25267671, -0.36428218
  );
  const mat3x3 m_inv = m.inverse();
  const mat3x3 m_inv_ref = new_mat3x3(
    -9.55800812, -11.30104242,  20.01798482,
    -4.47725118,  -9.5716761 ,  12.28271512,
    -2.27763488,  -5.6602941 ,   4.04056147
  );
  std::cout << m_inv << '\n' << m_inv_ref << std::endl;
}

void test_main() {
  const int wnd_size = cfg::WND_SIZE;
  RenderDevice<CubeSquareFractal>render_device(wnd_size, wnd_size);
  ProjectiveCamera cam;
  vec3 cam_position = new_vec3(3.f, 3.f, 3.f);
  vec3 right_vec = new_vec3(-1.f,1.f,0.f).normalized_unsafe() * 1.5;
  vec3 up_vec = new_vec3(-1.,-1.,2.).normalized_unsafe() * 1.5;
  cam.setupPersp(cam_position, right_vec, up_vec,
      2.f, 4.f
  );
  cam.applyWorldTransform(new_mat4x4_rotation(0.f, 0.f, 1.f));
  // cam.setupPersp(
      // new_vec3(0.f, 0.f, -3.f), new_vec3(-1.f, 0.f, 0.f), new_vec3(0.f, 1.f, 0.f),
      // 1.f, 5.f
  // );
  // const int num_iter = cfg::RT_SAMPLE_MULT;
  printf("preparing depth cache ..."); fflush(stdout);
  render_device.prepareDepthCache();
  printf(" done\n");
  render_device.setCamera(cam);
  render_device.initRayBuffer(false, false);
  render_device.renderDepthCache();
  printf("depth cache size:%d\n", render_device.depth_cache_size);

  const int num_iter = cfg::RT_SAMPLE_MULT;
  for(int i=0; i<num_iter; ++i) {
    render_device.initRayBuffer(true, true);
    render_device.render(true);
    std::cout << '.' << std::flush;
  }
  std::cout << std::endl;

  render_device.scaleFramebuffer(1.f/num_iter);
  render_device.exportFramebufferToFile("a.raw");
}
void test_mat4x4_rotation() {
  mat4x4 m = new_mat4x4_rotation(new_vec3(0.5, 0.3, -0.25));
  std::cout << m << std::endl;
}
void test_vec3_cross() {
  vec3 vx = new_vec3(-1.f, 1.f, 0.f);
  vec3 vy = new_vec3(-1.f,-1.f, 1.f);
  std::cout
    << cross(vx,vy) << '\n'
    << cross(vy,vx) << std::endl;
}
} // namespace unittest
#endif

int main(int argc, char** argv) {
  // unittest::mat3x3_inverse();
  unittest::test_main();
  return 0;
}
