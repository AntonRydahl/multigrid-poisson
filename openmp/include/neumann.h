#ifndef NEUMANN_BOUNDARY
#define NEUMANN_BOUNDARY

#include "boundary.h"

namespace Poisson{
    template <class T>
    class Neumann :
        public Boundary<T> {
            public:
            Neumann(int_t device,Location_t location,uint_t i, uint_t j, uint_t k);

            virtual ~Neumann();

            void init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings);

            void write_to(DeviceArray<T> & uarr, Settings & settings);

            void update(DeviceArray<T> & uarr, Settings & settings);

            void restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction);

            bool is_internal_boundary();

            void init_zero();

            void to_device();

            void link(Boundary<T> * _boundary);
    };

    template<class T>
    Neumann<T>::Neumann(int_t device,Location_t location,uint_t i, uint_t j, uint_t k) 
        : Boundary<T>(device,location,i,j,k){

    };

    template<class T>
    Neumann<T>::~Neumann() {
        // Does nothing
    }

    template<class T>
    void Neumann<T>::init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings){
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
                cerr << "Neumann conditions are not yet supported on TOP and BOTTOM" << endl; 
                break;
        }
    }

    template<class T>
    void Neumann<T>::write_to(DeviceArray<T> & uarr, Settings & settings){
        int_t offx = uarr.halo.west;
        int_t offy = uarr.halo.south;
        int_t offz = uarr.halo.bottom;

        int_t ii = 0;
        int_t jj = 0;
        int_t kk = 0;

        T sign = 1.0;

        switch (this->location){
            case EAST:
                offx = uarr.shape[0]-1+uarr.halo.east+uarr.halo.west;
                ii = -2;
                break;
            case WEST:
                offx = 0;
                ii = 2;
                sign = -1.0;
                break;
            case NORTH:
                offy = uarr.shape[1]-1+uarr.halo.north+uarr.halo.south;
                jj = -2;
                break;
            case SOUTH:
                offy=0;
                sign = -1.0;
                jj = 2;
                break;
            case TOP:
                offz = uarr.shape[2]-1+uarr.halo.top+uarr.halo.bottom;
                kk = -2;
                break;
            case BOTTOM:
                offz = 0;
                sign = -1.0;
                kk = 2;
                break;
        }
        T * udev = uarr.devptr;
        T * gdev = this->arr.devptr;
        const uint_t (&ustride)[3] = uarr.stride;
        const T two_h = 2.0*settings.h;
        const uint_t (&_shape)[3] = this->arr.shape;
        const uint_t (&_stride)[3] = this->arr.stride;
        const Halo & _halo = this->arr.halo;
        #pragma omp target device(this->arr.device) is_device_ptr(udev,gdev) firstprivate(sign,two_h)
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
                        udev[idx_halo(i+offx,j+offy,k+offz,ustride)] =
                            sign*two_h*gdev[idx(i,j,k,_halo,_stride)]+udev[idx_halo(i+offx+ii,j+offy+jj,k+offz+kk,ustride)];
#ifdef BLOCK_SIZE
                    }
#endif
                }
            }
        }
    };

    template<class T>
    void Neumann<T>::update(DeviceArray<T> & uarr, Settings & settings){
        #pragma omp task default(none) shared(uarr,settings) depend(inout:uarr)
        {
            this->write_to(uarr,settings);
        }
    }

    template<class T>
    void Neumann<T>::restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction){
        // Needs to be implemented
        restriction.restrict_to(boundary.arr,this->arr);
        int_t offx = 0;
        int_t offy = 0;
        int_t offz = 0;

        int_t ii = 0;
        int_t jj = 0;
        int_t kk = 0;
        int_t iii = 0;
        int_t jjj = 0;
        int_t kkk = 0;

        switch (this->location){
            case EAST:
                offx = ((uint_t) u.shape[0]/2)+1-1;
                ii = -1;
                iii = -2;
                break;
            case WEST:
                offx = 0;
                ii = 1;
                iii = 2;
                break;
            case NORTH:
                offy = ((uint_t) u.shape[1]/2)+1-1;
                jj = -1;
                jjj = -2;
                break;
            case SOUTH:
                offy=0;
                jj = 1;
                jjj = 2;
                break;
            case TOP:
                offz = ((uint_t) u.shape[2]/2)+1-1;
                kk = -1;
                kkk = -2;
                break;
            case BOTTOM:
                offz = 0;
                kk = 1;
                kkk = 2;
                break;
        }

        // Polynomail coefficients
        //gx1c = gx1 - (-(3/2)*unew(1,:,:)+2*unew(2,:,:)-0.5*unew(3,:,:))./h;
        //gxnc = gxn - (0.5*unew(end-2,:,:)-2*unew(end-1,:,:)+(3/2)*unew(end,:,:))./h;
        T c1 = -3.0/2.0;
        T c2 = 2.0;
        T c3 = -0.5;
        switch (this->location){
            case EAST:
            case NORTH:
            case TOP:
                c3 = 0.5;
                c2 = -2.0;
                c1 = 3.0/2.0;
                break;
            default:
                break;
        }

        // if (this->location == EAST){
        //     cout << "this->shape: (" << this->shape[0] << "," << this->shape[1] << "," << this->shape[2] << ")" << endl;
        //     cout << "u-shape:     (" << u.shape[0] << "," << u.shape[1] << "," << u.shape[2] << ")" << endl;
        //     cout << "Offsets:     (" << offx << "," << offy << "," << offz << ")" << endl;
        //     cout << c1 << " * u[i] + " << c2 << " * u[i+(" << ii << ")] + " << c3 << " * u[i+(" << iii << ")]" << endl;
        // }

        // Interpolating the Neumann condition
        T h = settings.h;
        T * udev = u.devptr;
        T * gdev = this->arr.devptr;
        const Halo & uhalo = u.halo;
        const uint_t (&ustride)[3] = u.stride;
        const uint_t * _shape = this->arr.shape;
        const Halo & _halo = this->arr.halo;
        const uint_t (&_stride)[3] = this->arr.stride;
        const uint _device = this->arr.device;
        #pragma omp target device(_device) is_device_ptr(udev,gdev)\
                firstprivate(c1,c2,c3,h,ii,jj,kk,iii,jjj,kkk)
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
                        gdev[idx(i,j,k,_halo,_stride)] -= 
                            (c1 * udev[idx(2*(i+offx),2*(j+offy),2*(k+offz),uhalo,ustride)] +
                            c2 * udev[idx(2*(i+offx)+ii,2*(j+offy)+jj,2*(k+offz)+kk,uhalo,ustride)] +
                            c3 * udev[idx(2*(i+offx)+iii,2*(j+offy)+jjj,2*(k+offz)+kkk,uhalo,ustride)])/h;
#ifdef BLOCK_SIZE
                    }
#endif
                }
            }
        }
    }

    template<class T>
    bool Neumann<T>::is_internal_boundary(){
        return false;
    };

    template<class T>
    void Neumann<T>::init_zero(){
        this->arr.init_zero();
    };

    template<class T>
    void Neumann<T>::to_device(){
        this->arr.to_device();
    };

    template<class T>
    void Neumann<T>::link(Boundary<T> * _boundary){
        // No need to link as it is not an internal boundary
    };
}

#endif
