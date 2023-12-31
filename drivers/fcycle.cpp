#include "parser.h"
#include "domainsettings.h"
#include "libpoisson.h"
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;
using Poisson::PoissonSolver;
using Poisson::Settings;
using Poisson::parser;
using Poisson::FullWeighting;
using Poisson::Injection;
using Poisson::GaussSeidel;
using Poisson::TrilinearInterpolation;
using Poisson::Jacobi;
using Poisson::BoundaryCondition;

int main(int argc, char * argv[]){
    Settings settings = parser(argc,argv);
    BoundaryCondition BC;
    BC.east = NEUMANN;
    BC.west = NEUMANN;
    BC.north = NEUMANN;
    BC.south = NEUMANN;
    BC.top = DIRICHLET;
    BC.bottom = DIRICHLET;
    PoissonSolver<double_t,Injection,TrilinearInterpolation,Jacobi> solver(settings,BC);

    solver.init();
    solver.verbose(true);
    solver.to_device();

    solver.solve("fcycle",4);
    cout << "It took " << solver.solve_time() << " seconds to run ";
    cout << solver.solve_iterations() << " Fcycles"<<endl;
    
    if (settings.print_result){
        solver.to_host();
        solver.save_all("results/u_vcycle.vtk","results/f_vcycle.vtk","results/r_vcycle.vtk");
    }

    if (settings.write_final_stats){
        ofstream out("results/"+settings.stats_file, ios::app);
        out << "#    seconds     rel_res domain_size     spacing     maxiter     lengthx      levels" << endl;
        out << setw(12) << solver.solve_time();
        out << setw(12) << solver.relative_residual();
        out << setw(12) << settings.dims[0]*settings.dims[1]*settings.dims[2];
        out << setw(12) << settings.h;
        out << setw(12) << solver.solve_iterations();
        out << setw(12) << settings.lengthx;
        out << setw(12) << settings.levels;
        out << endl;
    }

    return EXIT_SUCCESS;
}
