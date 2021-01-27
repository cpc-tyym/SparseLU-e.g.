#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>


using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double, RowMajor> spmat;
typedef Eigen::Triplet<double> T;



int main()
{

    // read file
    ifstream inFile1("d://A.csv", ios::in);
    ifstream inFile2("d://b.csv", ios::in);

    string lineStr;
    vector<T> triplets;

    int i, j;
    double tmpval;
    i = 0;
    char* end;
    if (inFile1.fail())
        cout << "reading failed" << endl;

    // get matrix A from csv file
    while (getline(inFile1, lineStr))
    {
        j = 0;
        stringstream ss(lineStr);
        string str;
        while (getline(ss, str, ','))
        {   
            tmpval = static_cast<double>(strtod(str.c_str(), &end));
            if (tmpval != 0) { triplets.push_back(T(i, j, tmpval)); }
            j++;
        }
        i++;
    }
    spmat A(i, j);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();


    // get vector b from csv file
    Eigen::VectorXd b(i);
    i = 0;
    while (getline(inFile2, lineStr))
    {
        j = 0;
        stringstream ss(lineStr);
        string str;
        while (getline(ss, str, ','))
        {
            b(j) = static_cast<double>(strtod(str.c_str(), &end));
            j++;
        }
        i++;
    }


    // solve the linear system: Ax = b
    Eigen::VectorXd sparselu_x, pardisolu_x;
    SparseLU<spmat> solver_splu;
    PardisoLU<spmat> solver_pdslu;

    // SparseLU results
    solver_splu.compute(A);
    sparselu_x = solver_splu.solve(b);

    // PardisoLU results
    solver_pdslu.compute(A);
    pardisolu_x = solver_pdslu.solve(b);


    // print some of the difference
    cout << "sparselu results - pardisolu results (last 5 items):\n\n" << (sparselu_x - pardisolu_x).tail(5) << endl;

    cout << "\n\n\n\nA*x - b (sparselu results, last 5 items):\n\n" << (A * sparselu_x - b).tail(5) << endl;
    cout << "\n\n\n\nA*x - b (pardisolu results, last 5 items):\n\n" << (A * pardisolu_x - b).tail(5) << endl;

    return 0;
}