#pragma once
#include "ray.h"
#include "mat.h"

class Material;

struct hit_record;

class Hitable 
{
public:
    __device__ virtual bool hit(const Ray& r, GLfloat t_min, GLfloat t_max, hit_record& rec) const = 0;
};
