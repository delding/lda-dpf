package ding.del.lda;

import java.io.File;

public class GibbsSampler {
  LDAModel trnModel; // model to be trained

  public GibbsSampler(LDAModel model) {
    trnModel = model;
  }

  public void setModel(LDAModel model) {
    trnModel = model;
  }

  public void initialize() {
    trnModel.initialize();
  }

  public void initFromFile(String nwFile) {
    trnModel.initFromFile(nwFile);
  }

  public void estimate() {
    System.out.println("Sampling " + trnModel.options.iterationNum + " iterations!");
    for (int i = 0; i < trnModel.options.iterationNum; ++i) {
      System.out.println("Iteration " + i + " ...");

      if ((i >= trnModel.options.saveInterval) &&
              ((i - trnModel.options.burnIn) % trnModel.options.saveInterval == 0)) {
        System.out.println("Saving the model at iteration " + i + " ...");
        computeTheta();
        computePhi();
//        trnModel.saveModel(trnModel.options.modelName + "-iteration" + i);
      }

      for (int d = 0; d < trnModel.D; d++) {
        for (int n = 0; n < trnModel.corpus.docs.get(d).length; n++) {
          int newTopic = gibbsSampling(d, n);
          trnModel.z[d][n] = newTopic;
        }
      }
    }

    System.out.println("Gibbs gibbsSampling completed!\n");
    System.out.println("Saving the final model!\n");
    computeTheta();
    computePhi();
//    trnModel.saveModel(trnModel.options.modelName + "-final");
    trnModel.saveNw(trnModel.options.dir + File.separator + "trainedNw");
    trnModel.saveNd(trnModel.options.dir + File.separator + "trainedNd");
  }

  /**
   * Do gibbsSampling
   *
   * @param d document number
   * @param n word number
   * @return topic assignment index
   */
  private int gibbsSampling(int d, int n) {
    // remove z_i from the count variable
    int topic = trnModel.z[d][n];
    int w = trnModel.corpus.docs.get(d).words[n];
    trnModel.nw[w][topic] -= 1;
    trnModel.nd[d][topic] -= 1;
    trnModel.nwsum[topic] -= 1;
    trnModel.ndsum[d] -= 1;

    double Vbeta = trnModel.V * trnModel.beta;
    double Kalpha = trnModel.K * trnModel.alpha;

    // Compute p(z_i = k | z_-i, w) via collapsed Gibbs gibbsSampling.
    double[] p = new double[trnModel.K];
    for (int k = 0; k < trnModel.K; k++) {
      p[k] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + Vbeta) *
              (trnModel.nd[d][k] + trnModel.alpha) / (trnModel.ndsum[d] + Kalpha);
    }

    // Do multinomial gibbsSampling for a new topic assignment of w_{m, n}
    // from Mult(p[0], p[1], ... , p[K-1]).
    // Compute cumulative probability for p.
    for (int k = 1; k < trnModel.K; k++) {
      p[k] += p[k - 1];
    }

    // scaled sample because of unnormalized p[]
    double u = Math.random() * p[trnModel.K - 1];

    // If topic = K - 2 still no break, topic goes to K - 1, loop ends
    for (topic = 0; topic < trnModel.K - 1; topic++) {
      if (p[topic] > u)
        break;
    }

    // Add newly estimated topic assignment z_i to count variables.
    trnModel.nw[w][topic] += 1;
    trnModel.nd[d][topic] += 1;
    trnModel.nwsum[topic] += 1;
    trnModel.ndsum[d] += 1;

    return topic;
  }

  public void computeTheta() {
    for (int m = 0; m < trnModel.D; m++) {
      for (int k = 0; k < trnModel.K; k++) {
        trnModel.theta[m][k] = (trnModel.nd[m][k] + trnModel.alpha)
            / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
      }
    }
  }

  public void computePhi() {
    for (int k = 0; k < trnModel.K; k++) {
      for (int w = 0; w < trnModel.V; w++) {
        trnModel.phi[k][w] = (trnModel.nw[w][k] + trnModel.beta)
            / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
      }
    }
  }
}