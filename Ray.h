#pragma once
#include "Vector.h"

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const vec3& a, const vec3& b) : A(a), B(b) {};
    __device__ vec3 origin() const 
    { 
        return A;
    }
    __device__ vec3 direction() const
    {
        return B;
    }
    __device__ vec3 point_at_parameter(GLfloat t) const
    {
        return A + t * B;
    }

    vec3 A;
    vec3 B;
};
