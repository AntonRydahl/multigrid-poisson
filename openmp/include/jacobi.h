#ifndef POISSON_JACOBI
#define POISSON_JACOBI

#include "definitions.h"
#include "domain.h"
#include "devicearray.h"
#include "relaxation.h"

#include <iostream>

namespace Poisson{
    template <typename T>
    class Domain;

    using std::cout;
    using std::endl;

    template <class T>
    class Jacobi : 
        public Relaxation<T> {
            public:
                void relax(Domain<T> *** domain,T omega,int_t nsmooth,int_t level,int_t num_devices);
                constexpr T default_omega();
                constexpr bool requires_duplicate_solution();
                void relaxation_kernel(Domain<T>& domain,T omega,const int_t xmin,const int_t xmax,const int_t ymin,const int_t ymax,const int_t zmin,const int_t zmax);
                void fill_send_buffer(Domain<T>& domain, const T omega, Location_t loc);
    };

    template <class T>
    void Jacobi<T>::relax(Domain<T> *** domains,T omega,int_t nsmooth,int_t level,int_t num_devices){
        
        //#pragma omp parallel
        //#pragma omp single nowait
        //#pragma omp taskgroup
        {
            for (int_t s = 0;s<nsmooth;s++){
                // Swapping domains
                //#pragma omp taskgroup
                {
                #pragma omp parallel for num_threads(num_devices)
                for (int_t gpuid=0;gpuid<num_devices;gpuid++){
                    Domain<T> & domain = *domains[gpuid][level];
                    #pragma omp task default(none) shared(domain) depend(inout:domain.uprev->at[0],domain.u->at[0])
                    domain.swap_u();

                    domains[gpuid][level]->east->update(*domains[gpuid][level],true);
                    domains[gpuid][level]->west->update(*domains[gpuid][level],true);
                    domains[gpuid][level]->north->update(*domains[gpuid][level],true);
                    domains[gpuid][level]->south->update(*domains[gpuid][level],true);
                    domains[gpuid][level]->top->update(*domains[gpuid][level],true);
                    domains[gpuid][level]->bottom->update(*domains[gpuid][level],true);
                }

                // Computing halo values
                #pragma omp parallel for num_threads(num_devices)
                for (int_t gpuid=0;gpuid<num_devices;gpuid++){
                    fill_send_buffer(*domains[gpuid][level],omega,EAST);
                    fill_send_buffer(*domains[gpuid][level],omega,WEST);
                    fill_send_buffer(*domains[gpuid][level],omega,NORTH);
                    fill_send_buffer(*domains[gpuid][level],omega,SOUTH);
                    fill_send_buffer(*domains[gpuid][level],omega,TOP);
                    fill_send_buffer(*domains[gpuid][level],omega,BOTTOM);
                    
                    Domain<T> & domain = *domains[gpuid][level];
                    DeviceArray<T>& u = *domain.u;

                    const int_t xmin = domain.west->is_non_eliminated();
                    const int_t xmax = u.shape[0]-domain.east->is_non_eliminated();
                    const int_t ymin = domain.south->is_non_eliminated();
                    const int_t ymax = u.shape[1]-domain.north->is_non_eliminated();
                    const int_t zmin = domain.bottom->is_non_eliminated();
                    const int_t zmax = u.shape[2]-domain.top->is_non_eliminated();
                    #pragma omp task default(none) shared(domain,omega)\
                    firstprivate(xmin,xmax,ymin,ymax,zmin,zmax)\
                    depend(inout:domain.u->at[0],domain.uprev->at[0]) \
                    depend(in:domain.east->arr.at[0],domain.west->arr.at[0],domain.north->arr.at[0],\
                    domain.south->arr.at[0],domain.top->arr.at[0],domain.bottom->arr.at[0])
                    this->relaxation_kernel(domain,omega,xmin,xmax,ymin,ymax,zmin,zmax);
                }
                }
            }
        }
    }

    template <class T>
    constexpr T Jacobi<T>::default_omega(){
        return 6.0/7.0;
    }

    template <class T>
    constexpr bool Jacobi<T>::requires_duplicate_solution(){
        return true;
    }

    template <class T>
    void Jacobi<T>::relaxation_kernel(Domain<T>& domain,T omega,const int_t xmin,const int_t xmax,const int_t ymin,const int_t ymax,const int_t zmin,const int_t zmax){
        DeviceArray<T>& u = *domain.u;
        DeviceArray<T>& v = *domain.uprev;
        const DeviceArray<T>& f = *domain.f;
        const T one_omega = 1.0-omega;
        const T omega_sixth = (omega/6.0);
        const T hsq = domain.settings.h*domain.settings.h;
        T * udev = u.devptr;
        T * vdev = v.devptr;
        T * fdev = f.devptr;

        const Halo & uhalo = u.halo;
        const uint_t (&ustride)[3] = u.stride;

        const Halo & vhalo = v.halo;
        const uint_t (&vstride)[3] = v.stride;

        const Halo & fhalo = f.halo;
        const uint_t (&fstride)[3] = f.stride;

        #pragma omp target device(u.device) is_device_ptr(udev,vdev,fdev)
        {
            #pragma omp teams distribute parallel for collapse(3) SCHEDULE DIST_SCHEDULE
            for (int_t i = xmin;i<xmax;i++){
                for (int_t j = ymin;j<ymax;j++){
#ifdef BLOCK_SIZE
                    for (int_t k_block = zmin;k_block<zmax;k_block+=BLOCK_SIZE){
                        #pragma omp simd
                        for (int_t k = k_block;k<MIN(k_block+BLOCK_SIZE,zmax);k++){
#else
                    for (int_t k = zmin;k<zmax;k++){
#endif
                            udev[idx(i,j,k,uhalo,ustride)] = one_omega*vdev[idx(i,j,k,vhalo,vstride)];
                            udev[idx(i,j,k,uhalo,ustride)] += omega_sixth*(vdev[idx((i-1),j,k,vhalo,vstride)] + vdev[idx((i+1),j,k,vhalo,vstride)]
                                                    +vdev[idx(i,(j-1),k,vhalo,vstride)] + vdev[idx(i,(j+1),k,vhalo,vstride)]
                                                    +vdev[idx(i,j,(k-1),vhalo,vstride)] + vdev[idx(i,j,(k+1),vhalo,vstride)] - hsq*fdev[idx(i,j,k,fhalo,fstride)]);
#ifdef BLOCK_SIZE
                        }
#endif
                    }
                }
            }
        }
    }

    template <class T>
    void Jacobi<T>::fill_send_buffer(Domain<T>& domain, const T omega, Location_t loc)
    {
        Boundary<T> * bound;
        switch(loc){
            case NORTH:
                bound = domain.north;
                break;
            case SOUTH:
                bound = domain.south;
                break;
            case EAST:
                bound = domain.east;
                break;
            case WEST:
                bound = domain.west;
                break;
            case TOP:
                bound = domain.top;
                break;
            case BOTTOM:
                bound = domain.bottom;
                break;
        }
        if (bound->is_internal_boundary()){
            OMPBoundary<T> * omp_bound = (OMPBoundary<T> *) bound;
            #pragma omp task default(none) shared(domain,omega) firstprivate(omp_bound) depend(in:bound->arr.at[0]) depend(out:omp_bound->send_buffer.at[0])
            omp_bound->fill_send_buffer(*domain.uprev,*domain.f,domain.settings,domain,omega);        
        }
    }
}
#endif
