
#include "hitable.h"
#include "mat.h"

class Sphere : public Hitable {
public:
    __device__ Sphere() {}
    __device__ Sphere(vec3 cen, GLfloat r, Material *mat) : center(cen), radius(r), mat(mat) {};
    __device__ virtual bool hit(const Ray& r, GLfloat tmin, GLfloat tmax, hit_record& rec) const;

    vec3 center;
    GLfloat radius;
    Material *mat;
};

__device__ bool Sphere::hit(const Ray& r, GLfloat t_min, GLfloat t_max, hit_record& rec) const
{
    vec3 oc = r.origin() - center;
    GLfloat a = dot(r.direction(), r.direction());
    GLfloat b = dot(oc, r.direction());
    GLfloat c = dot(oc, oc) - radius * radius;

    GLfloat discriminant = b * b - a * c;

    if (discriminant > 0) 
    {
        GLfloat temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            return true;
        }
    }
    return false;
}
