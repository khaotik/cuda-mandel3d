#pragma once
#include <iostream>
#include "common.cuh"

namespace curand {
#include <curand_kernel.h>
}

struct vec3; CUDA_FN vec3 new_vec3(float,float,float);
struct vec4;
  CUDA_FN vec4 new_vec4(float,float,float,float);
  CUDA_FN vec4 new_vec4(const vec3&, float);
union mat3x3;
  CUDA_FN mat3x3 new_mat3x3(const vec3&, const vec3&, const vec3&);
  CUDA_FN mat3x3 new_mat3x3(
      float, float, float, float, float, float, float, float, float);
  CUDA_FN mat3x3 new_mat3x3(float);
  CUDA_FN mat3x3 new_mat3x3_ident();
  CUDA_FN mat3x3 new_mat3x3_linear(const vec3&, const vec3&, const vec3&);
  CUDA_FN mat3x3 new_mat3x3_rotation(const vec3&);
  CUDA_FN mat3x3 new_mat3x3_rotation(float,float,float);
union mat4x4;
  CUDA_FN mat4x4 new_mat4x4(const vec4&, const vec4&, const vec4&, const vec4&);
  CUDA_FN mat4x4 new_mat4x4(
      float,float,float,float,float,float,float,float,
      float,float,float,float,float,float,float,float);
  CUDA_FN mat4x4 new_mat4x4(float);
  CUDA_FN mat4x4 new_mat4x4_diag(const vec4&);
  CUDA_FN mat4x4 new_mat4x4_ident();
  CUDA_FN mat4x4 new_mat4x4_translation(const vec3&);
  CUDA_FN mat4x4 new_mat4x4_linear(const vec3&, const vec3&, const vec3&);
  CUDA_FN mat4x4 new_mat4x4_rotation(const vec3&);
  CUDA_FN mat4x4 new_mat4x4_rotation(float,float,float);

CUDA_FN float dot(const vec3&, const vec3&);
CUDA_FN float dot(const vec4&, const vec3&);
CUDA_FN float dot(const vec4&, const vec4&);
CUDA_FN vec3 dot(const mat3x3&, const vec3&);
CUDA_FN vec3 dot(const mat4x4&, const vec3&);
CUDA_FN vec4 dot(const mat4x4&, const vec4&);
CUDA_FN vec3 dot(const vec3&, const mat3x3&);
CUDA_FN vec4 dot(const vec4&, const mat4x4&);
CUDA_FN mat3x3 dot(const mat3x3&, const mat3x3&);
CUDA_FN mat4x4 dot(const mat4x4&, const mat4x4&);

// gotta use factory here, gcc forbid anon structs in a union with constructors
// np.allclose
CUDA_FN bool allclose(float x, float y, float rtol=1e-5f, float atol=1e-8f) {
  return fabsf(x-y) <= (atol + rtol*fabsf(y));
}

struct vec3 {
  float x,y,z;
  CUDA_FN vec3 operator - () const { return new_vec3(-x, -y, -z); }
  CUDA_FN vec3 operator + (const vec3 &other) const { return new_vec3(x+other.x, y+other.y, z+other.z); }
  CUDA_FN vec3 operator - (const vec3 &other) const { return new_vec3(x-other.x, y-other.y, z-other.z); }
  CUDA_FN vec3 operator * (const vec3 &other) const { return new_vec3(x*other.x, y*other.y, z*other.z); }
  CUDA_FN vec3 operator * (const float scalar) const { return new_vec3(x*scalar, y*scalar, z*scalar); }
  CUDA_FN vec3 operator / (const float scalar) const {
    float iscalar = 1.f / scalar;
    return (*this) * iscalar;
  }
  CUDA_FN void operator += (const vec3 &other) { (*this) = (*this) + other; }
  CUDA_FN void operator -= (const vec3 &other) { (*this) = (*this) - other; }
  CUDA_FN void operator *= (const float scalar) { (*this) = (*this) * scalar; }
  CUDA_FN void operator /= (const float scalar) { (*this) = (*this) / scalar; }
  CUDA_FN float norm_unsafe() const {
    return sqrtf(x*x + y*y + z*z);
  }
  CUDA_FN float norm() const {
    // TODO.opt still can optimize out one mult
    float x_abs = fabsf(x);
    float y_abs = fabsf(y);
    float z_abs = fabsf(z);
    float scale = x_abs>y_abs ? x_abs : y_abs;
    if (scale < z_abs) { scale = z_abs; }
    float iscale = 1.f / scale;
    float x_ = x*iscale, y_ = y*iscale, z_ = z*iscale;
    return sqrtf(x_*x_ + y_*y_ + z_*z_) * scale;
  }
  CUDA_FN vec3 normalized_unsafe() const {
    float inorm = rsqrtf(x*x + y*y + z*z);
    return new_vec3(x*inorm, y*inorm, z*inorm);
  }
  CUDA_FN vec3 normalized_unsafe(float &norm_value) const {
    norm_value = sqrtf(x*x + y*y + z*z);
    float inorm = 1.f / norm_value;
    return new_vec3(x*inorm, y*inorm, z*inorm);
  }
  CUDA_FN vec3 normalized() const {
    // TODO.opt still can optimize out one mult
    float x_abs = fabsf(x), y_abs = fabsf(y), z_abs = fabsf(z);
    float scale = x_abs>y_abs ? x_abs : y_abs;
    if (scale < z_abs) { scale = z_abs; }
    float iscale = 1.f / scale;
    float x_ = x*iscale, y_ = y*iscale, z_ = z*iscale;
    float inorm = rsqrtf(x_*x_ + y_*y_ + z_*z_) * iscale;
    return new_vec3(x*inorm, y*inorm, z*inorm);
  }
  CUDA_FN vec3 normalized(float &norm_value) const {
    // TODO.opt still can optimize out one mult
    float x_abs = fabsf(x), y_abs = fabsf(y), z_abs = fabsf(z);
    float scale = x_abs>y_abs ? x_abs : y_abs;
    if (scale < z_abs) { scale = z_abs; }
    float iscale = 1.f / scale;
    float x_ = x*iscale, y_ = y*iscale, z_ = z*iscale;
    norm_value = sqrtf(x_*x_ + y_*y_ + z_*z_)*scale;
    float inorm = rsqrtf(x_*x_ + y_*y_ + z_*z_) * iscale;
    return new_vec3(x*inorm, y*inorm, z*inorm);
  }
};
CUDA_FN vec3 new_vec3(float x, float y=0.f, float z=0.f) {
  vec3 ret{x, y, z}; return ret;
}
CUDA_FN bool allclose(const vec3 &vX, const vec3 &vY) {
  return allclose(vX.x,vY.x) && allclose(vX.y,vY.y) && allclose(vX.z,vY.z);
}
CUDA_FN unsigned char float_to_uchar(float x) {
  return (unsigned char)(x*255.f);
}

struct vec4 {
  float x,y,z,w;
  CUDA_FN vec4 operator - () const { return new_vec4(-x, -y, -z, -w); }
  CUDA_FN vec4 operator + (const vec4 &other) const { return new_vec4(x+other.x, y+other.y, z+other.z, w+other.w); }
  CUDA_FN vec4 operator - (const vec4 &other) const { return new_vec4(x-other.x, y-other.y, z-other.z, w-other.w); }
  CUDA_FN vec4 operator * (const float scalar)const { return new_vec4(x*scalar, y*scalar, z*scalar, w*scalar); }
  CUDA_FN vec4 operator * (const vec4 &other) const { return new_vec4(x*other.x, y*other.y, z*other.z, w*other.w); }
  CUDA_FN vec4 operator / (const float scalar) const {
    float iscalar = 1.f / scalar;
    return (*this) * iscalar;
  }
  CUDA_FN void operator += (const vec4 &other) { (*this) = (*this) + other; } 
  CUDA_FN void operator -= (const vec4 &other) { (*this) = (*this) - other; } 
  CUDA_FN void operator *= (const float scalar) { (*this) = (*this) * scalar; } 
  CUDA_FN void operator /= (const float scalar) { (*this) = (*this) / scalar; } 
  CUDA_FN operator vec3() const {
    vec3 ret = new_vec3(x,y,z); return ret;
  }
  CUDA_FN operator uchar4() const {
    uchar4 ret{ float_to_uchar(x), float_to_uchar(y), float_to_uchar(z), float_to_uchar(w) };
    return ret;
  }
};
CUDA_FN vec4 new_vec4(float x, float y=0.f, float z=0.f, float w=0.f) {
  vec4 ret{x, y, z, w}; return ret;
}
CUDA_FN vec4 new_vec4(const vec3 &v, float w=1.f) {
  vec4 ret{v.x, v.y, v.z, w}; return ret;
}
CUDA_FN bool allclose(const vec4 &vX, const vec4 &vY) {
  return allclose(vX.x,vY.x) && allclose(vX.y,vY.y) && allclose(vX.z,vY.z) && allclose(vX.w,vY.w);
}

union mat3x3 {
  struct {vec3 rX, rY, rZ;};
  struct {float
    xx, xy, xz,
    yx, yy, yz,
    zx, zy, zz;
  };
  CUDA_FN void initConst(float value = 0.f) {
    rX = new_vec3(value, value, value);
    rY = new_vec3(value, value, value);
    rZ = new_vec3(value, value, value);
  }
  CUDA_FN void initDiag(const vec3 &diag = new_vec3(1.f,1.f,1.f)) {
    rX = new_vec3(diag.x, 0.f, 0.f);
    rY = new_vec3(0.f, diag.y, 0.f);
    rZ = new_vec3(0.f, 0.f, diag.z);
  }
  CUDA_FN void addDiag(float d=1.f) {
    xx += d; yy += d; zz += d;
  }
  CUDA_FN vec3 cX() const { return new_vec3(rX.x, rY.x, rZ.x); }
  CUDA_FN vec3 cY() const { return new_vec3(rX.y, rY.y, rZ.y); }
  CUDA_FN vec3 cZ() const { return new_vec3(rX.z, rY.z, rZ.z); }
  CUDA_FN mat3x3 operator *(float scalar) {
    return new_mat3x3(rX*scalar, rY*scalar, rZ*scalar);
  }
  CUDA_FN mat3x3 adjugate() const {
    float dxyxy = xx*yy - xy*yx; float dxyxz = xx*yz - xz*yx; float dxyyz = xy*yz - xz*yy;
    float dxzxy = xx*zy - xy*zx; float dxzxz = xx*zz - xz*zx; float dxzyz = xy*zz - xz*zy;
    float dyzxy = yx*zy - yy*zx; float dyzxz = yx*zz - yz*zx; float dyzyz = yy*zz - yz*zy;
    return new_mat3x3(
      dyzyz,-dxzyz, dxyyz,
     -dyzxz, dxzxz,-dxyxz,
      dyzxy,-dxzxy, dxyxy
    );
  }
  CUDA_FN mat3x3 inverse() const {
    float dxyxy = xx*yy - xy*yx; float dxyxz = xx*yz - xz*yx; float dxyyz = xy*yz - xz*yy;
    float dxzxy = xx*zy - xy*zx; float dxzxz = xx*zz - xz*zx; float dxzyz = xy*zz - xz*zy;
    float dyzxy = yx*zy - yy*zx; float dyzxz = yx*zz - yz*zx; float dyzyz = yy*zz - yz*zy;
    float idet = 1.f / (dxyxy*zz - dxyxz*zy + dxyyz*zx);
    return new_mat3x3(
      dyzyz,-dxzyz, dxyyz,
     -dyzxz, dxzxz,-dxyxz,
      dyzxy,-dxzxy, dxyxy
    ) * idet;
  }
  CUDA_FN mat3x3 T() const {
    return new_mat3x3(
        xx, yx, zx,
        xy, yy, zy,
        xz, yz, zz );
  }
  CUDA_FN mat3x3 normalized_l0() {
    // TODO.feat
    float l0norm = fabsf(xx), x;
    x = fabsf(xy); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(xz); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(yx); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(yy); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(yz); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(zx); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(zy); l0norm = l0norm>x ? l0norm : x;
    x = fabsf(zz); l0norm = l0norm>x ? l0norm : x;
    float scale = 1.f/l0norm;
    return new_mat3x3(rX*scale, rY*scale, rZ*scale);
  }
};
CUDA_FN mat3x3 new_mat3x3(
    const vec3 &rX, const vec3 &rY, const vec3 &rZ) {
  mat3x3 ret;
  ret.rX = rX; ret.rY = rY; ret.rZ = rZ;
  return ret;
}
CUDA_FN mat3x3 new_mat3x3(
  float xx, float xy, float xz,
  float yx, float yy, float yz,
  float zx, float zy, float zz ) {
  mat3x3 ret;
  ret.xx=xx; ret.xy=xy; ret.xz=xz;
  ret.yx=yx; ret.yy=yy; ret.yz=yz;
  ret.zx=zx; ret.zy=zy; ret.zz=zz;
  return ret;
}
CUDA_FN mat3x3 new_mat3x3(float x=0.f) {
  return new_mat3x3( x,x,x, x,x,x, x,x,x );
}
CUDA_FN mat3x3 new_mat3x3_diag(const vec3 &diag=new_vec3(1.f, 1.f, 1.f)) {
  mat3x3 ret;
  ret.initDiag(diag);
  return ret;
}
CUDA_FN mat3x3 new_mat3x3_ident() {
  mat3x3 ret;
  ret.initDiag();
  return ret;
}
CUDA_FN mat3x3 new_mat3x3_linear(const vec3& u, const vec3& v, const vec3& w) {
  return new_mat3x3(u,v,w).T();
}
CUDA_FN mat3x3 new_mat3x3_rotation(const vec3& full_v) {
  float radians, c, s;
  vec3 v = full_v.normalized_unsafe(radians);
  sincosf(radians, &s, &c); c = 1.f - c;
  vec3 vs = new_vec3(-v.x, v.y, -v.z)*s;
  vec3 v2 = v*v;
  vec3 vD = new_vec3(v2.y+v2.z, v2.x+v2.z, v2.x+v2.y) * (-c) + new_vec3(1.f,1.f,1.f);
  vec3 vc= new_vec3(
      v.y*v.z,
      v.x*v.z,
      v.x*v.y) * c;
  vec3 vU = vc + vs;
  vec3 vL = vc - vs;
  return new_mat3x3(
      vD.x, vU.z, vU.y,
      vL.z, vD.y, vU.x,
      vL.y, vL.x, vD.z
  );
}
CUDA_FN mat3x3 new_mat3x3_rotation(float x, float y, float z) {
  return new_mat3x3_rotation(new_vec3(x,y,z));
}

union mat4x4 {
  struct {vec4 rX, rY, rZ, rW;};
  struct {float
    xx, xy, xz, xw,
    yx, yy, yz, yw,
    zx, zy, zz, zw,
    wx, wy, wz, ww;
  };
  CUDA_FN void initConst(float value = 0.f) {
    rX = new_vec4(value, value, value, value);
    rY = new_vec4(value, value, value, value);
    rZ = new_vec4(value, value, value, value);
    rW = new_vec4(value, value, value, value);
  }
  CUDA_FN void initDiag(const vec4 &diag = new_vec4(1.f,1.f,1.f,1.f)) {
    rX = new_vec4(diag.x, 0.f, 0.f, 0.f);
    rY = new_vec4(0.f, diag.y, 0.f, 0.f);
    rZ = new_vec4(0.f, 0.f, diag.z, 0.f);
    rW = new_vec4(0.f, 0.f, 0.f, diag.w);
  }
  CUDA_FN void addDiag(float d=1.f) {
    xx += d; yy += d; zz += d; ww += d;
  }
  CUDA_FN vec4 cX() const { return new_vec4(rX.x, rY.x, rZ.x, rW.x); }
  CUDA_FN vec4 cY() const { return new_vec4(rX.y, rY.y, rZ.y, rW.y); }
  CUDA_FN vec4 cZ() const { return new_vec4(rX.z, rY.z, rZ.z, rW.z); }
  CUDA_FN vec4 cW() const { return new_vec4(rX.w, rY.w, rZ.w, rW.w); }
  CUDA_FN mat4x4 operator *(float scalar) {
    return new_mat4x4(rX*scalar, rY*scalar, rZ*scalar, rW*scalar);
  }
  CUDA_FN mat4x4 adjugate() const {
    float
    s0 = xx*yy - yx*xy, s1 = xx*yz - yx*xz,
    s2 = xx*yw - yx*xw, s3 = xy*yz - yy*xz,
    s4 = xy*yw - yy*xw, s5 = xz*yw - yz*xw,
    c5 = zz*ww - wz*zw, c4 = zy*ww - wy*zw,
    c3 = zy*wz - wy*zz, c2 = zx*ww - wx*zw,
    c1 = zx*wz - wx*zz, c0 = zx*wy - wx*zy;
    return new_mat4x4(
       yy*c5-yz*c4+yw*c3,-xy*c5+xz*c4-xw*c3, wy*s5-wz*s4+ww*s3,-zy*s5+zz*s4-zw*s3,
      -yx*c5+yz*c2-yw*c1, xx*c5-xz*c2+xw*c1,-wx*s5+wz*s2-ww*s1, zx*s5-zz*s2+zw*s1,
       yx*c4-yy*c2+yw*c0,-xx*c4+xy*c2-xw*c0, wx*s4-wy*s2+ww*s0,-zx*s4+zy*s2-zw*s0,
      -yx*c3+yy*c1-yz*c0, xx*c3-xy*c1+xz*c0,-wx*s3+wy*s1-wz*s0, zx*s3-zy*s1+zz*s0
    );
  }
  CUDA_FN mat4x4 inverse() const {
    float
    s0 = xx*yy - yx*xy, s1 = xx*yz - yx*xz,
    s2 = xx*yw - yx*xw, s3 = xy*yz - yy*xz,
    s4 = xy*yw - yy*xw, s5 = xz*yw - yz*xw,
    c5 = zz*ww - wz*zw, c4 = zy*ww - wy*zw,
    c3 = zy*wz - wy*zz, c2 = zx*ww - wx*zw,
    c1 = zx*wz - wx*zz, c0 = zx*wy - wx*zy,
    idet = 1.f/(s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0);
    return new_mat4x4(
       yy*c5-yz*c4+yw*c3,-xy*c5+xz*c4-xw*c3, wy*s5-wz*s4+ww*s3,-zy*s5+zz*s4-zw*s3,
      -yx*c5+yz*c2-yw*c1, xx*c5-xz*c2+xw*c1,-wx*s5+wz*s2-ww*s1, zx*s5-zz*s2+zw*s1,
       yx*c4-yy*c2+yw*c0,-xx*c4+xy*c2-xw*c0, wx*s4-wy*s2+ww*s0,-zx*s4+zy*s2-zw*s0,
      -yx*c3+yy*c1-yz*c0, xx*c3-xy*c1+xz*c0,-wx*s3+wy*s1-wz*s0, zx*s3-zy*s1+zz*s0
    )*idet;
  }
  CUDA_FN mat4x4 T() const {
    return new_mat4x4(
        xx, yx, zx, wx,
        xy, yy, zy, wy,
        xz, yz, zz, wz,
        xw, yw, zw, ww );
  }
};
CUDA_FN mat4x4 new_mat4x4(
    const vec4 &rX, const vec4 &rY, const vec4 &rZ, const vec4 &rW) {
  mat4x4 ret;
  ret.rX = rX; ret.rY = rY; ret.rZ = rZ; ret.rW = rW;
  return ret;
}
CUDA_FN mat4x4 new_mat4x4(
  float xx, float xy, float xz, float xw, float yx, float yy, float yz, float yw,
  float zx, float zy, float zz, float zw, float wx, float wy, float wz, float ww) {
  mat4x4 ret;
  ret.xx=xx; ret.xy=xy; ret.xz=xz; ret.xw=xw; ret.yx=yx; ret.yy=yy; ret.yz=yz; ret.yw=yw;
  ret.zx=zx; ret.zy=zy; ret.zz=zz; ret.zw=zw; ret.wx=wx; ret.wy=wy; ret.wz=wz; ret.ww=ww;
  return ret;
}
CUDA_FN mat4x4 new_mat4x4(float x=0.f) {
  return new_mat4x4( x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x );
}
CUDA_FN mat4x4 new_mat4x4_diag(const vec4 &diag=new_vec4(1.f, 1.f, 1.f, 1.f)) {
  mat4x4 ret;
  ret.initDiag(diag);
  return ret;
}
CUDA_FN mat4x4 new_mat4x4_ident() {
  mat4x4 ret;
  ret.initDiag();
  return ret;
}
CUDA_FN mat4x4 new_mat4x4_translation(const vec3 &v) {
  return new_mat4x4(
    1.f, 0.f, 0.f, v.x,
    0.f, 1.f, 0.f, v.y,
    0.f, 0.f, 1.f, v.z,
    0.f, 0.f, 0.f, 1.f);
}
CUDA_FN mat4x4 new_mat4x4_rotation(const vec3 &full_v) {
  /* return expm([
       [ 0,-z, y],
       [ z, 0,-x],
       [-y, x, 0]
     ]) via rodrigues formula
  */
  // TODO.refac code duplication from new_mat3x3_rotation
  float radians, c, s;
  vec3 v = full_v.normalized_unsafe(radians);
  sincosf(radians, &s, &c); c = 1.f - c;
  vec3 vs = new_vec3(-v.x, v.y, -v.z)*s;
  vec3 v2 = v*v;
  vec3 vD = new_vec3(v2.y+v2.z, v2.x+v2.z, v2.x+v2.y) * (-c) + new_vec3(1.f,1.f,1.f);
  vec3 vc= new_vec3(
      v.y*v.z,
      v.x*v.z,
      v.x*v.y) * c;
  vec3 vU = vc + vs;
  vec3 vL = vc - vs;
  return new_mat4x4(
      vD.x, vU.z, vU.y, 0.f,
      vL.z, vD.y, vU.x, 0.f,
      vL.y, vL.x, vD.z, 0.f,
      0.f,  0.f,  0.f,  1.f
  );
}
CUDA_FN mat4x4 new_mat4x4_rotation(float x,float y,float z) {
  return new_mat4x4_rotation(new_vec3(x,y,z));
}
CUDA_FN mat4x4 new_mat4x4_linear(const vec3 &u, const vec3 &v, const vec3 &w) {
  return new_mat4x4(
      new_vec4(u, 0.f), new_vec4(v, 0.f), new_vec4(w, 0.f),
      new_vec4(0.f, 0.f, 0.f, 1.f)
  ).T();
}
CUDA_FN mat4x4 new_mat4x4_projection(float z_far, float z_scale) {
  // const float u = z_scale;
  const float v = (1.f/z_scale) - 1.f;
  const float w = z_far / z_scale;
  return new_mat4x4(
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, w, 0.f,
    0.f, 0.f, v, 1.f
  );
}
CUDA_FN bool allclose(const mat4x4 &mX, const mat4x4 &mY) {
  return allclose(mX.rX, mY.rX) && allclose(mX.rY, mY.rY) && allclose(mX.rZ, mY.rZ) && allclose(mX.rW, mY.rW) ;
}
CUDA_FN float dot(const vec3 &vx, const vec3 &vy) {
  return vx.x * vy.x + vx.y * vy.y + vx.z * vy.z;
}
CUDA_FN float dot(const vec4 &vx, const vec3 &vy) {
  return vx.x * vy.x + vx.y * vy.y + vx.z * vy.z + vx.w ;
}
CUDA_FN float dot(const vec4 &vx, const vec4 &vy) {
  return vx.x * vy.x + vx.y * vy.y + vx.z * vy.z + vx.w * vy.w ;
}
CUDA_FN vec3 dot(const mat3x3 &m, const vec3 &v) {
  return new_vec3(dot(m.rX, v), dot(m.rY, v), dot(m.rZ, v));
}
CUDA_FN vec3 dot(const mat4x4 &m, const vec3 &v) {
  // linear fractional transform
  return new_vec3(dot(m.rX, v), dot(m.rY, v), dot(m.rZ, v)) / dot(m.rW, v);
}
CUDA_FN vec4 dot(const mat4x4 &m, const vec4 &v) {
  return new_vec4(dot(m.rX, v), dot(m.rY, v), dot(m.rZ, v), dot(m.rW, v));
}
CUDA_FN vec4 dot(const vec4 &v, const mat4x4 &m) {
  return m.rX*v.x + m.rY*v.y + m.rZ*v.z + m.rW*v.w;
}
CUDA_FN vec3 dot(const vec3 &v, const mat3x3 &m) {
  return m.rX*v.x + m.rY*v.y + m.rZ*v.z;
}
CUDA_FN mat3x3 dot(const mat3x3 &m, const mat3x3 &n) {
  return new_mat3x3( dot(m.rX, n), dot(m.rY, n), dot(m.rZ, n) );
}
CUDA_FN mat4x4 dot(const mat4x4 &m, const mat4x4 &n) {
  return new_mat4x4( dot(m.rX, n), dot(m.rY, n), dot(m.rZ, n), dot(m.rW, n) );
}
CUDA_FN vec3 cross(const vec3 &vx, const vec3 &vy) {
  return new_vec3(
      vx.y*vy.z - vx.z*vy.y,
      vx.z*vy.x - vx.x*vy.z,
      vx.x*vy.y - vx.y*vy.x );
}
CUDA_FN vec3 project_unsafe(const vec3 &dst, const vec3 &src) {
  return dst*(dot(dst, src) / dot(dst, dst));
}

namespace unittest {
std::ostream& operator <<(std::ostream &os, const vec3 &v) {
  return os << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}
std::ostream& operator <<(std::ostream &os, const vec4 &v) {
  return os << '[' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ']';
}
std::ostream& operator <<(std::ostream &os, const mat3x3 &m) {
  return os << '[' << m.rX << ',' << m.rY << ',' << m.rZ << ']';
}
std::ostream& operator <<(std::ostream &os, const mat4x4 &m) {
  return os << '[' << m.rX << ',' << m.rY << ',' << m.rZ << ',' << m.rW << ']';
}
template<typename T>
struct NpyAsArray {
  const std::string lhs;
  const T *rhs; // TODO.feat use a better smart pointer for safety?
  NpyAsArray(const std::string lhs_, const T &rhs_):lhs(lhs_),rhs(&rhs_) {}
};
template<typename T>
inline NpyAsArray<T> np_asarray(const std::string &lhs, const T &rhs) {
  return NpyAsArray<T>(lhs, rhs);
}
template<typename T>
inline std::ostream& operator << (std::ostream &os, const NpyAsArray<T> &expr) {
  os << expr.lhs << " = np.asarray(" << *(expr.rhs) << ')';
  return os;
}
}; // namespace unittest
