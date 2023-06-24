#include "problem_definition.h"
#ifdef PROBLEM1
namespace Poisson {
    #define kx 3.0
    #define ky 5.3
    #define kz -7.8

    double_t ufun(double_t x,double_t y,double_t z){
        return sin(kx*x)*sin(ky*y)*sin(kz*z);
    }

    double_t ffun(double_t x,double_t y,double_t z){
        return -(kx*kx+ky*ky+kz*kz)*sin(kx*x)*sin(ky*y)*sin(kz*z);
    }

    double_t dudxfun(double_t x,double_t y,double_t z){
        return kx*cos(kx*x)*sin(ky*y)*sin(kz*z);
    }

    double_t dudyfun(double_t x,double_t y,double_t z){
        return ky*sin(kx*x)*cos(ky*y)*sin(kz*z);
    }

    double_t dudzfun(double_t x,double_t y,double_t z){
        return kz*sin(kx*x)*sin(ky*y)*cos(kz*z);
    }
}
#endif
#ifdef PROBLEM2 
namespace Poisson {
    #define k0 ((double) 5.0)

    double_t ufun(double_t x,double_t y,double_t z){
        return x*x*x*y*y*z;
    }

    double_t ffun(double_t x,double_t y,double_t z){
        return 2.0*x*x*x*z + 6.0*x*y*y*z;
    }

    double_t dudxfun(double_t x,double_t y,double_t z){
        return 3.0*x*x*y*y*z;
    }

    double_t dudyfun(double_t x,double_t y,double_t z){
        return 2.0*x*x*x*y*z;
    }

    double_t dudzfun(double_t x,double_t y,double_t z){
        return x*x*x*y*y;
    }
}
#endif
#ifdef PROBLEM3
namespace Poisson {
    double_t ufun(double_t x,double_t y,double_t z){
        return cos(x*z*z)*sin(y*y*y);
    }

    double_t ffun(double_t x,double_t y,double_t z){
        return ((-4.0*x*x*z*z - 9.0*y*y*y*y - z*z*z*z)*sin(y*y*y) + 6.0*y*cos(y*y*y))*cos(x*z*z) - 2.0*x*sin(x*z*z)*sin(y*y*y);
    }

    double_t dudxfun(double_t x,double_t y,double_t z){
        return -z*z*sin(x*z*z)*sin(y*y*y);
    }

    double_t dudyfun(double_t x,double_t y,double_t z){
        return 3.0*cos(x*z*z)*y*y*cos(y*y*y);
    }

    double_t dudzfun(double_t x,double_t y,double_t z){
        return -2*x*z*sin(x*z*z)*sin(y*y*y);
    }
}
#endif
