// Spectral clustering, by Tim Nugent 2014

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <map>
#include <math.h>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "spectral.h"

using namespace Eigen;
using namespace std;

int Spectral::read_data(const char* data, char sep){

	// Read data
	unsigned int row = 0;
	ifstream file(data);
	if(file.is_open()){
		string line,token;
		while(getline(file, line)){
			stringstream tmp(line);
			unsigned int col = 0;
			while(getline(tmp, token, sep)){
				if(X.rows() < row+1){
					X.conservativeResize(row+1,X.cols());
				}
				if(X.cols() < col+1){
					X.conservativeResize(X.rows(),col+1);
				}
				X(row,col) = atof(token.c_str());
				col++;
			}
			row++;
		}
		file.close();
		return(1);
	}else{
		return(0);
	}
}

int Spectral::write_data(const char* data, char sep){

	// Read data
	unsigned int row = 0;
	ofstream file(data);
	if(file.is_open()){
		for(unsigned int i = 0; i < X.rows(); i++){
			for(unsigned int j = 0; j < X.cols(); j++){
				file << X(i,j) << sep;
			}
			file << assignments[row] << endl;
			row++;
		}	
		file.close();
		return(1);
	}else{
		return(0);
	}
}

double Spectral::kernel(const VectorXd& a, const VectorXd& b){

	switch(kernel_type){
	    case 2  :
	    	return(pow(a.dot(b)+constant,order));
	    default : 
	    	return(exp(-gamma*((a-b).squaredNorm())));
	}

}

void Spectral::generate_kernel_matrix(){

	// Fill kernel matrix
	K.resize(X.rows(),X.rows());
	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = i; j < X.rows(); j++){
			K(i,j) = K(j,i) = kernel(X.row(i),X.row(j));
			//if(i == 0) cout << K(i,j) << " ";
		}	
	}

	// Normalise kernel matrix	
	VectorXd d = K.rowwise().sum();
	for(unsigned int i = 0; i < d.rows(); i++){
		d(i) = 1.0/sqrt(d(i));
	}
	auto F = d.asDiagonal();
	MatrixXd l = (K * F);
	for(unsigned int i = 0; i < l.rows(); i++){
		for(unsigned int j = 0; j < l.cols(); j++){
			l(i,j) = l(i,j) * d(i);
		}
	}		
	K = l;

}

void Spectral::eigendecomposition(){

	EigenSolver<MatrixXd> edecomp(K);
	eigenvalues = edecomp.eigenvalues().real();
	eigenvectors = edecomp.eigenvectors().real();
	cumulative.resize(eigenvalues.rows());
	vector<pair<double,VectorXd> > eigen_pairs; 
	double c = 0.0; 
	for(unsigned int i = 0; i < eigenvectors.cols(); i++){
		if(normalise){
			double norm = eigenvectors.col(i).norm();
			eigenvectors.col(i) /= norm;
		}
		eigen_pairs.push_back(make_pair(eigenvalues(i),eigenvectors.col(i)));
	}
	// http://stackoverflow.com/questions/5122804/sorting-with-lambda
	sort(eigen_pairs.begin(),eigen_pairs.end(), [](const pair<double,VectorXd> a, const pair<double,VectorXd> b) -> bool {return (a.first > b.first);} );

	if(centers > eigen_pairs.size()) centers = eigen_pairs.size();

	for(unsigned int i = 0; i < eigen_pairs.size(); i++){	
		eigenvalues(i) = eigen_pairs[i].first;
		c += eigenvalues(i);
		cumulative(i) = c;
		eigenvectors.col(i) = eigen_pairs[i].second;
	}

	/*
	cout << "Sorted eigenvalues:" << endl;
	for(unsigned int i = 0; i < eigenvalues.rows(); i++){
		if(eigenvalues(i) > 0){
			cout << "PC " << i+1 << ": Eigenvalue: " << eigenvalues(i);
			printf("\t(%3.3f of variance, cumulative =  %3.3f)\n",eigenvalues(i)/eigenvalues.sum(),cumulative(i)/eigenvalues.sum());
			//cout << eigenvectors.col(i) << endl;
		}
	}
	cout << endl;
	*/
	MatrixXd tmp = eigenvectors;
	
	// Select top K eigenvectors where K = centers
	eigenvectors = tmp.block(0,0,tmp.rows(),centers);

}	


void Spectral::cluster(){

	generate_kernel_matrix();
	eigendecomposition();
	kmeans();

}

// Code adapted from https://github.com/pthimon/clustering
void Spectral::kmeans(){

	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> rand_index(0,eigenvectors.rows()-1);
	MatrixXd centroids = MatrixXd::Zero(centers, eigenvectors.cols());
	MatrixXd old_centroids;

	// Randomly select unique centroids 
	vector<int> rands;
	while(rands.size() < centers){
		int r = rand_index(gen);
		bool tag = false;
		for (unsigned int j=0; j < rands.size(); ++j){
			if(r == rands[j]){
				tag = true;
				break;
			}
		}
		if(!tag){				
			centroids.row(rands.size()) = eigenvectors.row(r);
			rands.push_back(r);	
		}
	}

	MatrixXd id = MatrixXd::Identity(centers, centers);
	VectorXd minvals(eigenvectors.rows());
	// Matrix to map vectors to centroids
	MatrixXd post(eigenvectors.rows(), centers);

	int r, c;
	double old_e = 0;
	for (unsigned int n=0; n < max_iters; n++){
		old_centroids = centroids;

		// Calculate distances
		MatrixXd d2(eigenvectors.rows(), centers);
		for (unsigned int j = 0; j < centers; j++){
			for(int k=0; k < eigenvectors.rows(); k++) {
				d2(k,j) = (eigenvectors.row(k)-centroids.row(j)).squaredNorm();
			}
		}
		
		// Assign to nearest centroid
		for (unsigned int k = 0; k < eigenvectors.rows(); k++){
			// Get index of centroid
			minvals[k] = d2.row(k).minCoeff(&r, &c);
			// Set centroid
			post.row(k) = id.row(c);
		}

		// Adjust centeroids
		VectorXd num_points = post.colwise().sum();
		for(unsigned int j = 0; j < centers; j++){
			if(num_points(j) > 0) {
				MatrixXd s = MatrixXd::Zero(1,eigenvectors.cols());
				for(unsigned int k = 0; k < eigenvectors.rows(); k++){
					if(post(k,j) == 1){
						s += eigenvectors.row(k);
					}
				}
				centroids.row(j) = s/num_points[j];
			}
		}

		// Calculate error - total squared distance from centroids
		double e = minvals.sum();
		double ediff = fabs(old_e-e);
		double cdiff = (centroids-old_centroids).cwiseAbs().maxCoeff();
		printf("Iterations %i : Error %2.4f : Error delta %2.4f : Centroid movement %2.4f\n",n+1,e,ediff,cdiff);
		if(n && cdiff < numeric_limits<double>::epsilon() && ediff < numeric_limits<double>::epsilon()){
			break;
		}
		old_e = e;
	}

	map<int,int> data_to_cluster;
	for (unsigned int j = 0; j < centers; j++){
		for (int k=0; k < eigenvectors.rows(); k++) {
			if (post(k,j) == 1) {
				data_to_cluster[k] = j+1;
			}
		}
	}
	for (int k=0; k < eigenvectors.rows(); k++){
		assignments.push_back(data_to_cluster[k]);
	}

}
