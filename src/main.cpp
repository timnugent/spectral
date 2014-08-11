#include <iostream>
#include <vector>
#include "spectral.h"

using namespace std;

int main(){

    const char* spirals_in  = "data/spirals.csv";
    const char* spirals_out = "data/spirals_clustered.csv";

    Spectral* P = new Spectral();

    if(P->read_data(spirals_in)){
        cout << "Read " << spirals_in << endl;
    }else{
        cout << "Failed to read " << spirals_in << endl;
        delete P;
        return(-1);
    } 
    
    P->set_gamma(172.05); // Set RBF gamma param 
    P->set_centers(2); // Set K = 2 - number of centroids and number of eigenvectors
    P->cluster(); // Cluster using K-means

    if(P->write_data(spirals_out)){
        cout << "Wrote " << spirals_out << endl;
    }else{
        cout << "Failed to write " << spirals_out << endl;
        delete P;
        return(-1);
    } 

    cout << "Cluster assignments:" << endl;
    vector<int> assignments = P->get_assignments();
    for(auto i : assignments) cout << i << " ";
    cout << endl;    

    delete P;
    return(0);
}
