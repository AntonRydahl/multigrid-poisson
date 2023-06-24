#ifndef PROBLEM_DEFINITION
#define PROBLEM_DEFINITION

#include "definitions.h" 
#include <cmath>

namespace Poisson{
    double_t ufun(double_t x,double_t y,double_t z);

    double_t ffun(double_t x,double_t y,double_t z);

    double_t dudxfun(double_t x,double_t y,double_t z);

    double_t dudyfun(double_t x,double_t y,double_t z);

    double_t dudzfun(double_t x,double_t y,double_t z);

    typedef double_t (*funptr)(double_t x,double_t y,double_t z);
}
#endif
