#ifndef POISSON_RESIDUAL
#define POISSON_RESIDUAL

#include "definitions.h"
#include "domain.h"
#include "devicearray.h"

#include <iostream>

using std::cout;
using std::endl;

namespace Poisson{
    template <typename T>
    class Domain;

    template <class T>
    void residual(Domain<T>& domain,bool fetch_neighbor=true){
        DeviceArray<T>& u = *domain.u;
        DeviceArray<T>& r = *domain.r;
        const DeviceArray<T>& f = *domain.f;

        const T hsq = domain.settings.h*domain.settings.h;

        // Updating boundaries
        //#pragma omp taskgroup
        {
            domain.east->update(domain,false,fetch_neighbor);
            domain.west->update(domain,false,fetch_neighbor);
            domain.north->update(domain,false,fetch_neighbor);
            domain.south->update(domain,false,fetch_neighbor);
            domain.top->update(domain,false,fetch_neighbor);
            domain.bottom->update(domain,false,fetch_neighbor);
        }

        // Extracting device pointers

        const int_t xmin = domain.west->is_non_eliminated();
        const int_t xmax = u.shape[0]-domain.east->is_non_eliminated();
        const int_t ymin = domain.south->is_non_eliminated();
        const int_t ymax = u.shape[1]-domain.north->is_non_eliminated();
        const int_t zmin = domain.bottom->is_non_eliminated();
        const int_t zmax = u.shape[2]-domain.top->is_non_eliminated();

        ////#pragma omp task default(none) shared(u,r,f) firstprivate(hsq,xmin,xmax,ymin,ymax,zmin,zmax) depend(in:u) depend(out:r)
        {
            T * udev = u.devptr;
            T * rdev = r.devptr;
            T * fdev = f.devptr;
            const Halo & rhalo = r.halo;
            const uint_t (&rstride)[3] = r.stride;

            const Halo & fhalo = f.halo;
            const uint_t (&fstride)[3] = f.stride;

            const Halo & uhalo = u.halo;
            const uint_t (&ustride)[3] = u.stride;
            #pragma omp target device(u.device) is_device_ptr(udev,rdev,fdev) firstprivate(hsq)
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
                                rdev[idx(i,j,k,rhalo,rstride)] = fdev[idx(i,j,k,fhalo,fstride)] - (udev[idx((i-1),j,k,uhalo,ustride)] + udev[idx((i+1),j,k,uhalo,ustride)]
                                                +udev[idx(i,(j-1),k,uhalo,ustride)] + udev[idx(i,(j+1),k,uhalo,ustride)]
                                                +udev[idx(i,j,(k-1),uhalo,ustride)] + udev[idx(i,j,(k+1),uhalo,ustride)] - 6.0*udev[idx(i,j,k,uhalo,ustride)])/hsq;
#ifdef BLOCK_SIZE
                            }
#endif
                        }
                    }
                }
            }
        }
    }
}
#endif
