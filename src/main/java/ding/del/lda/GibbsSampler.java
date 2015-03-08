package ding.del.lda;

public class GibbsSampler {
  LDAModel trnModel; // model to be trained

  public void initialize(LDAOptions options) {
    trnModel = new LDAModel();
    trnModel.initialize(options);
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
        trnModel.saveModel(trnModel.options.modelName + "-iteration" + i);
      }

      for (int m = 0; m < trnModel.D; m++) {
        for (int n = 0; n < trnModel.corpus.docs.get(m).length; n++) {
          int newTopic = gibbsSampling(m, n);
          trnModel.z[m][n] = newTopic;
        }
      }
    }

    System.out.println("Gibbs gibbsSampling completed!\n");
    System.out.println("Saving the final model!\n");
    computeTheta();
    computePhi();
    trnModel.saveModel(trnModel.options.modelName + "-final");
  }

  /**
   * Do gibbsSampling
   *
   * @param m document number
   * @param n word number
   * @return topic assignment index
   */
  private int gibbsSampling(int m, int n) {
    // remove z_i from the count variable
    int topic = trnModel.z[m][n];
    int w = trnModel.corpus.docs.get(m).words[n];
    trnModel.nw[w][topic] -= 1;
    trnModel.nd[m][topic] -= 1;
    trnModel.nwsum[topic] -= 1;
    trnModel.ndsum[m] -= 1;

    double Vbeta = trnModel.V * trnModel.beta;
    double Kalpha = trnModel.K * trnModel.alpha;

    // Compute p(z_i = k | z_-i, w) via collapsed Gibbs gibbsSampling.
    double[] p = new double[trnModel.K];
    for (int k = 0; k < trnModel.K; k++) {
      p[k] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + Vbeta) *
              (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + Kalpha);
    }

    // Do multinomial gibbsSampling for a new topic assignment of w_{m, n}
    // from Mult(p[0], p[1], ... , p[K-1]).
    // Compute cumulative probability for p.
    for (int k = 1; k < trnModel.K; k++) {
      p[k] += p[k - 1];
    }

    // scaled sample because of unnormalized p[]
    double u = Math.random() * p[trnModel.K - 1];

    for (topic = 0; topic < trnModel.K; topic++) {
      if (p[topic] > u)
        break;
    }

    // Add newly estimated topic assignment z_i to count variables.
    trnModel.nw[w][topic] += 1;
    trnModel.nd[m][topic] += 1;
    trnModel.nwsum[topic] += 1;
    trnModel.ndsum[m] += 1;

    return topic;
  }

  public void computeTheta() {
    for (int m = 0; m < trnModel.D; m++) {
      for (int k = 0; k < trnModel.K; k++) {
        trnModel.theta[m][k] = (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
      }
    }
  }

  public void computePhi() {
    for (int k = 0; k < trnModel.K; k++) {
      for (int w = 0; w < trnModel.V; w++) {
        trnModel.phi[k][w] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
      }
    }
  }
}