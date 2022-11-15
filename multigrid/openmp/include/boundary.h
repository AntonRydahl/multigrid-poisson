#ifndef POISSON_BOUNDARY
#define POISSON_BOUNDARY

#include "devicearray.h"
#include "definitions.h"
#include "problem_definition.h"

typedef enum {NORTH,SOUTH,EAST,WEST,TOP,BOTTOM} Location_t;

template <class T>
class Boundary :
    public DeviceArray<T> {
    protected:
        Location_t location;
        void init_by_fun(funptr fun,Settings & settings){
            const double_t x0 = settings.origin[0];
            const double_t y0 = settings.origin[1];
            const double_t z0 = settings.origin[2];
            const double_t h = settings.h;
            double_t offx = 0;
            double_t offy = 0;
            double_t offz = 0;
            switch (this->location){
                case TOP:
                    offz = settings.dims[2]-1;
                    break;
                case NORTH:
                    offy = settings.dims[1]-1;
                    break;
                case EAST:
                    offx = settings.dims[0]-1;
                    break;
                default:
                    break;
            }
            for(int_t i = 0;i<this->shape[0];i++){
                for(int_t j = 0;j<this->shape[1];j++){
                    for(int_t k = 0;k<this->shape[2];k++){
                        this->at[this->idx(i,j,k)] = fun(x0+(((double_t)i)+offx)*h,y0+(((double_t)j)+offy)*h,z0+(((double_t)k)+offz)*h);
                    }
                }
            }
        }

        public:
        Boundary(int_t device, Location_t location,uint_t i, uint_t j, uint_t k) : DeviceArray<T>(device,i,j,k){
            this->location = location;
        }

        virtual ~Boundary() {
            
        }
        
        virtual void init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings);

        void write_to(DeviceArray<T> * uarr,Settings & settings){
            int_t offx = 0;
            int_t offy = 0;
            int_t offz = 0;
            switch (this->location){
                case TOP:
                    offz = settings.dims[2]-1;
                    break;
                case NORTH:
                    offy = settings.dims[1]-1;
                    break;
                case EAST:
                    offx = settings.dims[0]-1;
                    break;
                default:
                    break;
            }
            for(int_t i = 0;i<this->shape[0];i++){
                for(int_t j = 0;j<this->shape[1];j++){
                    for(int_t k = 0;k<this->shape[2];k++){
                        uarr->at[uarr->idx(i+offx,j+offy,k+offz)] = this->at[this->idx(i,j,k)];
                    }
                }
            }
        };
};

template <class T>
class Dirichlet :
    public Boundary<T> {
        public:
        Dirichlet(int_t device,Location_t location,uint_t i, uint_t j, uint_t k) : Boundary<T>(device,location,i,j,k){

        }

        virtual ~Dirichlet(){
            //
        }

        void init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings){
            this->init_by_fun(ufun,settings);
        }

};

template <class T>
class Neumann :
    public Boundary<T> {
        public:
        Neumann(int_t device,Location_t location,uint_t i, uint_t j, uint_t k) : Boundary<T>(device,location,i,j,k){

        }

        virtual ~Neumann() {
            //
        }

        void init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings){
            switch (this->location){
                case EAST:
                case WEST:
                    this->init_by_fun(dudxfun,settings);
                    break;
                case NORTH:
                case SOUTH:
                    this->init_by_fun(dudyfun,settings);
                    break;
                default:
                    break;
            }
        }
};

#endif