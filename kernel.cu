
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GL/freeglut.h"

#include <stdio.h>
#include <iostream>
#include "Ray.h"
#include "hitable.h"
#include "Sphere.h"
#include "hitables.h"
#include "Camera.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

GLint width = 800, height = 600;

const GLint block_size_x =8, block_size_y = 8;
const dim3 blocks(width / block_size_x + 1, height / block_size_y + 1);
const dim3 threads(block_size_x, block_size_y);


const GLint OBJECT_NUM = 10; // должен быть вида x^2 + 1
Hitable** d_list;
Hitable** d_world;
vec3* pixels;
Camera camera;


__global__ void AllocateObjects(Hitable** d_list, Hitable** d_world, GLint sqrt_obj_num)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        Material *mat;
        for (GLint i = 0; i < sqrt_obj_num; i++)
        {
            for (GLint j = 0; j < sqrt_obj_num; j++)
            {
                if (i % 2 == 0)
                    mat = new Lambert(vec3(0.5, 0.5, 0.125));
                else
                    mat = new Metallic(vec3(0.2, 0.8, 0.55), 0.5);

                *(d_list + i*sqrt_obj_num + j) = new Sphere(vec3(2.5f*i, 1 , 2.5f*j), 0.8f, mat);
            }
        }
        *(d_list + sqrt_obj_num*sqrt_obj_num) = new Sphere(vec3(0, -100.5f, -1.5), 100, new Metallic(vec3(0.8, 0.8, 0.8),0)); 
        *d_world = new Hitables(d_list, OBJECT_NUM);
    }
}

__global__ void DeleteObjects(Hitable** d_list, Hitable** d_world) {
    for (GLint i = 0; i < OBJECT_NUM; i++)
    {
        delete d_list[i];
        delete ((Sphere*)d_list[i])->mat;
    }
    delete* d_world;
}

__device__ vec3 Color(const Ray& r, Hitable** world)
{
    Ray new_ray = r;
    vec3 cur_fading = vec3(1.0, 1.0, 1.0);
    vec3 col(0, 0, 0);
    for (GLint i = 0; i < 20; i++)
    {
        hit_record rec;
        if ((*world)->hit(new_ray, 0.01f, FLT_MAX, rec)) // вместо FLT_MAX можно использовать максимальную видимость
        {
            Ray diff;
            vec3 fading;
            if (rec.mat->Diffuse(new_ray, rec, fading, diff)) 
            {
                cur_fading *= fading;
                new_ray = diff;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(new_ray.direction());
            GLfloat t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_fading * c;
        }
    }
    return vec3(0, 0, 0);
}

__global__ void Raytrace(vec3* pixels, GLint width, GLint height,
    vec3 eye,  GLfloat W, GLfloat H, GLfloat N, vec3 n, vec3 u, vec3 v,
    Hitable** world) 
{
    GLint i = threadIdx.x + blockIdx.x * blockDim.x;
    GLint j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    GLint pixel_index = i * height + j;


    GLfloat x = -W + (i * 2 * W) / (GLfloat)width;
    GLfloat y = -H + (j * 2 * H) / (GLfloat)height;
    vec3 direction(-N * n.x() + x * u.x() + y * v.x(), -N * n.y() + x * u.y() + y * v.y(),  -N * n.z() + x * u.z() + y * v.z());
    direction.make_unit_vector();

    Ray r(eye, direction);
    pixels[pixel_index] = Color(r, world);
}

void Reshape(GLint w, GLint h)
{    
    width = w;
    height = h;

    GLint num_pixels = w*h;
    size_t pixels_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaFree(pixels));
    checkCudaErrors(cudaMallocManaged((void**)&pixels, pixels_size)); 
}

void FillPixels()
{
    gluOrtho2D(0, width, 0, height);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            glColor3f(pixels[i* height + j].r(), pixels[i * height + j].g(), pixels[i * height + j].b());
            glRecti(i, j, i + 1, j + 1);
        }
    }
}

void Draw(Hitable** world)
{      

    // выставить параметры камеры
    GLfloat aspect = width / (GLfloat)height;
    GLfloat N = 0.1f;
    GLfloat tetha = 60 * (M_PI / 180.0f);
    GLfloat H = N * tan(tetha / 2.0f);
    GLfloat W = H * aspect;
    vec3 n = unit_vector(camera.Position - camera.View); // camDirs
    vec3 u = unit_vector(cross(camera.UpVector, n)); // X axis 
    vec3 v = unit_vector(cross(n, u));

    clock_t start, stop;
    start = clock();
    Raytrace << <blocks, threads >> > (pixels, width, height, camera.Position,
                                    W, H, N, n, u, v,
                                    world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    GLfloat timer_seconds = ((GLfloat)(stop - start)) / CLOCKS_PER_SEC;
    printf("%f seconds.\n", timer_seconds) ;

    FillPixels();    
}

void Keyboard(unsigned char key, GLint x, GLint y)
{
    GLfloat kSpeed = 0.5f;
    if (key == 'w' || key == 'W')
    {
        camera.MoveCamera(kSpeed);
    }
    if (key == 's' || key == 'S')
    {
        camera.MoveCamera(-kSpeed);
    }
    if (key == 'a' || key == 'A')
    {
        camera.MoveCameraLeftRight(kSpeed);
    }
    if (key == 'd' || key == 'D')
    {
        camera.MoveCameraLeftRight(-kSpeed);
    }
}

void MouseMove(GLint x, GLint y)
{
    camera.MouseView(width, height);
}

void Display(void)
{    
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    Draw(d_world);
    glutSwapBuffers();
}


int main(int argc, char* argv[])
{
    checkCudaErrors(cudaMalloc((void**)&d_list, OBJECT_NUM * sizeof(Hitable*)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hitable*)));

    AllocateObjects << <1, 1 >> > (d_list, d_world, sqrt(OBJECT_NUM));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    camera.SetCamera(-7.0f, 10.0f, -7.0f, // положение камеры
        sqrt(OBJECT_NUM), -2.0f, sqrt(OBJECT_NUM),                 // точка, в которую смотреть
        0.0f, 1.0f, 0.0f);	            // поворот

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("RT with CUDA");
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Display);
    glutKeyboardFunc(Keyboard);
    glutPassiveMotionFunc(MouseMove);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();

    // чистка после выхода
    DeleteObjects << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    cudaDeviceReset();
    return 0;
}