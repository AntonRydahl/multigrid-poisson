#include "parser.h"
#include "domainsettings.h"
#include "libpoisson.h"
#include <iostream>

using std::cout;
using std::endl;
using Poisson::PoissonSolver;
using Poisson::Settings;
using Poisson::parser;
using Poisson::Injection;

int main(int argc, char * argv[]){
    bool is_dirichlet = false;
    Settings settings = parser(argc,argv);
    PoissonSolver<double_t,Injection> solver(settings,is_dirichlet);

    solver.init();
    solver.verbose(true);
    solver.to_device();

    solver.solve("vcycle");
    cout << "It took " << solver.solve_time() << " seconds to run ";
    cout << solver.solve_iterations() << " Vcycles"<<endl;
    
    solver.to_host();
    solver.save_all("results/u_vcycle.vtk","results/f_vcycle.vtk","results/r_vcycle.vtk");

    return EXIT_SUCCESS;
}
