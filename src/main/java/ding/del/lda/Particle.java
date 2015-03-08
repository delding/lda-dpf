package ding.del.lda;

import java.util.ArrayList;

public class Particle {

  int numDocuments;
  public ArrayList<int[]> z; // each item contains the topic assignments of a document
  // Each item is an array representing topic proportion parameter of a document.
  // The dimension of array equals to K, and the K-th element eta^{K-1} always equals to 0
  public ArrayList<double[]> eta;
  public double alpha[]; // dynamic parameter with dimension equal to K - 1

  public Particle(ArrayList<int[]> z, ArrayList<double[]> eta, double alpha[]) {
    numDocuments = z.size();
    this.z = z;
    this.eta = eta;
    this.alpha = alpha;
  }
}