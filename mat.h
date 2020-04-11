#pragma once


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vector.h"
#include "Ray.h"
#include "hitable.h"

class Material;
struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    Material* mat;
};

__device__ vec3 Reflect(const vec3& v, const vec3& n) 
{
    return v - 2.0f * dot(v, n) * n;
}

class Material 
{
public:
    __device__ virtual bool Diffuse(const Ray& r_in, const hit_record& rec, vec3& fading, Ray& diff) const = 0;

    vec3 albedo;
};

class Lambert : public Material 
{
public:
    __device__ Lambert(const vec3& a) 
    { 
        albedo = a;
    }
    __device__ virtual bool Diffuse(const Ray& r_in, const hit_record& rec, vec3& fading, Ray& diff) const
    {
        vec3 target = rec.p + rec.normal;
        diff = Ray(rec.p, target - rec.p);
        fading = albedo;
        return true;
    }
};

class Metallic : public Material
{
public:
    __device__ Metallic(const vec3& a, GLfloat f) 
    {
        albedo = a;
        if (f < 1)
            fuzz = f;
        else
            fuzz = 1;
    }
    __device__ virtual bool Diffuse(const Ray& r_in, const hit_record& rec, vec3& fading, Ray& diff) const
    {
        vec3 reflected = Reflect(unit_vector(r_in.direction()), rec.normal);
        diff = Ray(rec.p, reflected + fuzz*vec3(0.3,0.3,0.3)); //* vec3(0.2,0.5,0.1));
        fading = albedo;
        return (dot(diff.direction(), rec.normal) > 0.0f);
    }
    GLfloat fuzz;
};
