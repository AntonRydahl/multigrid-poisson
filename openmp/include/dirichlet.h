#ifndef DIRICHLET_BOUNDARY
#define DIRICHLET_BOUNDARY

#include "boundary.h"
namespace Poisson{
    template <class T>
    class Dirichlet :
        public Boundary<T> {
            public:
            Dirichlet(int_t device,Location_t location,uint_t i, uint_t j, uint_t k);

            virtual ~Dirichlet();

            void init(funptr ufun,funptr dudxfun,funptr dudyfun,funptr dudzfun,Settings & settings);

            void write_to(DeviceArray<T> & uarr, Settings & settings);

            void update(Domain<T> & domain,bool previous,bool fetch_neighbor);

            void restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction);

            bool is_internal_boundary();

            void init_zero();

            void to_device();

            void link(Boundary<T> * _boundary);

            bool is_non_eliminated();
    };

    template <class T>
    Dirichlet<T>::Dirichlet(int_t device,Location_t location,uint_t i, uint_t j, uint_t k) 
        : Boundary<T>(device,location,i,j,k){

    }

    template <class T>
    Dirichlet<T>::~Dirichlet(){
        // Does nothing
    }

    template <class T>
    void Dirichlet<T>::init(funptr ufun,funptr dudxfun,funptr dudyfun,funptr dudzfun,Settings & settings){
        this->init_by_fun(ufun,settings);
    }

    template<class T>
    void Dirichlet<T>::write_to(DeviceArray<T> & uarr, Settings & settings){
        int_t offx = 0;
        int_t offy = 0;
        int_t offz = 0;
        switch (this->location){
            case TOP:
                offz = uarr.shape[2]-1;
                break;
            case NORTH:
                offy = uarr.shape[1]-1;
                break;
            case EAST:
                offx = uarr.shape[0]-1;
                break;
            default:
                break;
        }
        T * udev = uarr.devptr;
        T * gdev = this->arr.devptr;
        const Halo & uhalo = uarr.halo;
        const uint_t (&ustride)[3] = uarr.stride;
        const uint_t (&_shape)[3] = this->arr.shape;
        const uint_t (&_stride)[3] = this->arr.stride;
        const Halo & _halo = this->arr.halo;
        #pragma omp target device(this->arr.device) is_device_ptr(udev,gdev)
        #pragma omp teams distribute parallel for collapse(3) SCHEDULE
        for(int_t i = 0;i<_shape[0];i++){
            for(int_t j = 0;j<_shape[1];j++){
#ifdef BLOCK_SIZE
                for(int_t k_block = 0;k_block<_shape[2];k_block+=BLOCK_SIZE){
                    #pragma omp simd
                    for(int_t k = k_block;k<MIN(k_block+BLOCK_SIZE,_shape[2]);k++){
#else
                for(int_t k = 0;k<_shape[2];k++){
#endif
                        udev[idx(i+offx,j+offy,k+offz,uhalo,ustride)] = gdev[idx(i,j,k,_halo,_stride)];
#ifdef BLOCK_SIZE
                    }
#endif
                }
            }
        }
    };

    template<class T>
    void Dirichlet<T>::update(Domain<T> & domain,bool previous,bool fetch_neighbor){
        // The Dirichlet condition only needs to be computed once
        //this->write_to(uarr,settings);
    }

    template<class T>
    void Dirichlet<T>::restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction){
        // For a dirichlet type boundary condition the error is always 0
        // as the solution is exact on this type of boundary by definition.
        this->arr.init_zero();
    };

    template<class T>
    bool Dirichlet<T>::is_internal_boundary(){
        return false;
    };

    template<class T>
    void Dirichlet<T>::init_zero(){
        this->arr.init_zero();
    };

    template<class T>
    void Dirichlet<T>::to_device(){
        this->arr.to_device();
    };

    template<class T>
    void Dirichlet<T>::link(Boundary<T> * _boundary){
        // No need to link as it is not an internal boundary
    };

    template<class T>
    bool Dirichlet<T>::is_non_eliminated(){
        return true;
    };
}
#endif
