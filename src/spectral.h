// Spectral clustering, by Tim Nugent 2014

#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

class Spectral{

public:
	Spectral() : centers(2), kernel_type(1), normalise(1), max_iters(1000), gamma(0.001), constant(1.0), order(2.0) {}
	explicit Spectral(MatrixXd& d) : centers(2), kernel_type(1), normalise(1), max_iters(1000), gamma(0.001), constant(1.0), order(2.0) {X = d;}
	int read_data(const char* data, char sep = ',');
	int write_data(const char* data, char sep = ',');
	void set_centers(const unsigned int i){centers = i;};
	void set_kernel(const unsigned int i){kernel_type = i;};	
	void set_normalise(const unsigned int i){normalise = i;};
	void set_gamma(const double i){gamma = i;};
	void set_constant(const int i){constant = i;};
	void set_order(const double i){order = i;};
	void set_max_iters(const unsigned int i){max_iters = i;};
	void cluster();
	const vector<int> &get_assignments() const {return assignments;};
private:
	void generate_kernel_matrix();
	double kernel(const VectorXd& a, const VectorXd& b);
	void eigendecomposition();
	void kmeans();
	MatrixXd X, K, eigenvectors;
	VectorXd eigenvalues, cumulative;
	unsigned int centers, kernel_type, normalise, max_iters;
	double gamma, constant, order;
	vector<int> assignments;
};

#endif
