#ifndef OMP_BOUNDARY
#define OMP_BOUNDARY

#include "boundary.h"
namespace Poisson{
    template <class T>
    class OMPBoundary :
        public Boundary<T> {
            public:
            DeviceArray<T> send_buffer;
            
            OMPBoundary(int_t device,Location_t location,uint_t i, uint_t j, uint_t k);

            virtual ~OMPBoundary();

            void init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings);

            void write_to(DeviceArray<T> & uarr, Settings & settings);

            void fill_send_buffer(DeviceArray<T> & varr,const DeviceArray<T> & farr, Settings & settings,const T omega);

            void update(DeviceArray<T> & uarr, Settings & settings);

            void restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction);

            bool is_internal_boundary();

            void init_zero();

            void to_device();

            void link(Boundary<T> * _neighbor);

            bool is_non_eliminated();
    };

    template <class T>
    OMPBoundary<T>::OMPBoundary(int_t device,Location_t location,uint_t i, uint_t j, uint_t k) 
        : Boundary<T>(device,location,i,j,k), send_buffer(device,i,j,k){

    }

    template <class T>
    OMPBoundary<T>::~OMPBoundary(){
        // Does nothing
    }

    template <class T>
    void OMPBoundary<T>::init(funptr ufun,funptr dudxfun,funptr dudyfun,Settings & settings){
        this->init_zero();
    }

    template<class T>
    void OMPBoundary<T>::write_to(DeviceArray<T> & uarr, Settings & settings){
        int_t offx = uarr.halo.west;
        int_t offy = uarr.halo.south;
        int_t offz = uarr.halo.bottom;

        switch (this->location){
            case EAST:
                offx = uarr.shape[0]-1+uarr.halo.east+uarr.halo.west;
                break;
            case WEST:
                offx = 0;
                break;
            case NORTH:
                offy = uarr.shape[1]-1+uarr.halo.north+uarr.halo.south;
                break;
            case SOUTH:
                offy=0;
                break;
            case TOP:
                offz = uarr.shape[2]-1+uarr.halo.top+uarr.halo.bottom;
                break;
            case BOTTOM:
                offz = 0;
                break;
        }
        T * udev = uarr.devptr;
        T * gdev = this->arr.devptr;
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
                        udev[idx_halo(i+offx,j+offy,k+offz,ustride)] = gdev[idx(i,j,k,_halo,_stride)];
#ifdef BLOCK_SIZE
                    }
#endif
                }
            }
        }
    };

    template<class T>
    void OMPBoundary<T>::fill_send_buffer(DeviceArray<T> & varr,const DeviceArray<T> & farr, Settings & settings,const T omega){
        int_t offx = 0;
        int_t offy = 0;
        int_t offz = 0;
        switch (this->location){
            case TOP:
                offz = varr.shape[2]-2;
                break;
            case NORTH:
                offy = varr.shape[1]-2;
                break;
            case EAST:
                offx = varr.shape[0]-2;
                break;
            case BOTTOM:
                offz = 1;
                break;
            case SOUTH:
                offy = 1;
                break;
            case WEST:
                offx = 1;
                break;
            default:
                break;
        }
        T * vdev = varr.devptr;
        T * fdev = farr.devptr;
        T * gdev = this->send_buffer.devptr;
        const Halo & vhalo = varr.halo;
        const uint_t (&vstride)[3] = varr.stride;
        const Halo & fhalo = farr.halo;
        const uint_t (&fstride)[3] = farr.stride;
        const uint_t (&_shape)[3] = this->send_buffer.shape;
        const uint_t (&_stride)[3] = this->send_buffer.stride;
        const Halo & _halo = this->send_buffer.halo;

        const T one_omega = 1.0-omega;
        const T omega_sixth = (omega/6.0);
        const T hsq = settings.h*settings.h;

        #pragma omp target device(this->send_buffer.device) is_device_ptr(vdev,gdev,fdev)
        #pragma omp teams distribute parallel for collapse(3) SCHEDULE
        for(int_t ii = 0;ii<_shape[0];ii++){
            for(int_t jj = 0;jj<_shape[1];jj++){
#ifdef BLOCK_SIZE
                for(int_t k_block = 0;k_block<_shape[2];k_block+=BLOCK_SIZE){
                    #pragma omp simd
                    for(int_t kk = k_block;kk<MIN(k_block+BLOCK_SIZE,_shape[2]);kk++){
#else
                for(int_t kk = 1;kk<_shape[2]-1;kk++){
#endif                  
                        int_t i = ii + offx;
                        int_t j = jj + offy;
                        int_t k = kk + offz;
                        gdev[idx(ii,jj,kk,_halo,_stride)] = one_omega*vdev[idx(i,j,k,vhalo,vstride)];
                        gdev[idx(ii,jj,kk,_halo,_stride)] += omega_sixth*(vdev[idx((i-1),j,k,vhalo,vstride)] + vdev[idx((i+1),j,k,vhalo,vstride)]
                                                    +vdev[idx(i,(j-1),k,vhalo,vstride)] + vdev[idx(i,(j+1),k,vhalo,vstride)]
                                                    +vdev[idx(i,j,(k-1),vhalo,vstride)] + vdev[idx(i,j,(k+1),vhalo,vstride)] - hsq*fdev[idx(i,j,k,fhalo,fstride)]);
#ifdef BLOCK_SIZE
                    }
#endif
                }
            }
        }
    };

    template<class T>
    void OMPBoundary<T>::update(DeviceArray<T> & uarr, Settings & settings){
        //cout << "Device is " << this->arr.device << endl;
        OMPBoundary<T> * omp_neighbor = ((OMPBoundary<T> * )this->neighbor);
        //#pragma omp task default(none) firstprivate(omp_neighbor) depend(in:omp_neighbor->send_buffer) depend(out:this->arr)
        {
            omp_neighbor->send_buffer.device_to_device(this->arr);
        }
        //#pragma omp task default(none) shared(uarr,settings) depend(in:this->arr) depend(in:uarr)
        {
            this->write_to(uarr,settings);
        }
    }

    template<class T>
    void OMPBoundary<T>::restrict_to(DeviceArray<T> & u, Boundary<T> & boundary,
                            Settings & settings, Restriction<T> & restriction){
        // The internal boundary can be thought of as a Dirichlet boundary which 
        // is updated in every iteration of the relaxation method. Hence, its residual
        // is always zero.
        this->arr.init_zero();
        this->send_buffer.init_zero();
    };

    template<class T>
    bool OMPBoundary<T>::is_internal_boundary(){
        return true;
    };

    template<class T>
    void OMPBoundary<T>::init_zero(){
        this->arr.init_zero();
        this->send_buffer.init_zero();
    };

    template<class T>
    void OMPBoundary<T>::to_device(){
        this->arr.to_device();
        this->send_buffer.to_device();
    };

    template<class T>
    void OMPBoundary<T>::link(Boundary<T> * _neighbor){
        this->neighbor = _neighbor;
    }

    template<class T>
    bool OMPBoundary<T>::is_non_eliminated(){
        return false;
    };
}
#endif
