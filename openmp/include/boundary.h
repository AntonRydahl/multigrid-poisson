#ifndef POISSON_BOUNDARY
#define POISSON_BOUNDARY

#include "devicearray.h"
#include "definitions.h"
#include "problem_definition.h"
#include "settings.h"
#include "halo.h"
#include "restriction.h"

namespace Poisson{
    typedef enum {NORTH,SOUTH,EAST,WEST,TOP,BOTTOM} Location_t;

    typedef enum {DIRICHLET,NEUMANN} Boundary_t;

    bool requires_halo(Boundary_t bound,int idx,int num_devices){
        if ((idx < num_devices -1) || (idx > 0)) {
            return true;
        }
        return bound != DIRICHLET;
    }

    typedef struct BoundaryCondition{
        Boundary_t north;
        Boundary_t south;
        Boundary_t east;
        Boundary_t west;
        Boundary_t top;
        Boundary_t bottom;
    } BoundaryCondition;

    template <typename T>
	class Domain;

    template <class T>
    class Boundary /*: public DeviceArray<T>*/ {
        protected:
            // The neighbor bounday is only needed for internal boundaries
            Boundary<T> * neighbor;
            Location_t location;
            void init_by_fun(funptr fun,Settings & settings);

            public:
            DeviceArray<T> arr;

            Boundary(int_t device, Location_t location,uint_t i, uint_t j, uint_t k);

            virtual ~Boundary();
            
            virtual void init(funptr ufun,funptr dudxfun,funptr dudyfun,funptr dudzfun,Settings & settings) = 0;

            virtual void write_to(DeviceArray<T> & uarr, Settings & settings) = 0;

            virtual void update(Domain<T> & domain,bool previous = false, bool fetch_neighbor=true) = 0;

            virtual void restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                                    Settings & settings, Restriction<T> & restriction) = 0;

            virtual bool is_internal_boundary() = 0;

            virtual void init_zero() = 0;

            virtual void to_device() = 0;

            virtual void link(Boundary<T> * _boundary) = 0;

            virtual bool is_non_eliminated() = 0;
    };

    template<class T>
    Boundary<T>::Boundary(int_t device, Location_t location,uint_t i, uint_t j, uint_t k) :
        arr(device,i,j,k) {
        this->location = location;
    }

    template<class T>
    Boundary<T>::~Boundary() {
        // Does nothing
    }

    template<class T>
    void Boundary<T>::init_by_fun(funptr fun,Settings & settings){
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
        #pragma omp parallel for collapse(3) SCHEDULE
        for(int_t i = 0;i<this->arr.shape[0];i++){
            for(int_t j = 0;j<this->arr.shape[1];j++){
                for(int_t k = 0;k<this->arr.shape[2];k++){
                    this->arr.at[this->arr.idx(i,j,k)] = fun(x0+(((double_t)i)+offx)*h,y0+(((double_t)j)+offy)*h,z0+(((double_t)k)+offz)*h);
                }
            }
        }
    }
}
#endif
