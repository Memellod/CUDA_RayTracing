#pragma once
#include "hitable.h"

class Hitables : public Hitable 
{
public:
    __device__ Hitables() {}
    __device__ Hitables(Hitable** l, GLint n)
    {
        list = l;
        list_size = n;
    }
    __device__ virtual bool hit(const Ray& r, GLfloat tmin, GLfloat tmax, hit_record& rec) const;
    Hitable** list;
    GLint list_size;
};

__device__ bool Hitables::hit(const Ray& r, GLfloat t_min, GLfloat t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    GLfloat closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) 
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
