package ding.del.lda;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

public class ParticleFilter {
  LDACorpus corpus =  null; // a single observation in this corpus contains a mini-batch of documents
  LDAOptions options;

  int V;
  int K = 100;
  double beta = 0.1;

  int windowSize; // length of sliding window
  int numParticles;

  // Each item in the list is an array of Particles, the length of array is numParticle.
  // This array contains particles given the observation of the documents in one time slice
  LinkedList<ArrayList<Particle>> particlesQueue;

  // Each array in the queue is the resample index for particles in that time slice
  LinkedList<int[]> resampleIndexQueue;

  // length = numParticles, each item is the weight of a particle path
  double[] weights;

  // size = numParticles
  // Each item in the list is a topic word counts matrix corresponding to particle path
  // nw[i][j] denotes number of word/term i assigned to topic j, size V * K
  // nwsum[j] denotes total number of words assigned to topic j, size K
  LinkedList<ArrayList<int[][]>> nwQueue;
  LinkedList<ArrayList<int[]>> nwsumQueue;

  ArrayList<Document> observe = null; // current observation

  Random rand = new Random();

  public ParticleFilter(int windowSize, int numParticles) {
    this.windowSize = windowSize;
    this.numParticles = numParticles;
    particlesQueue = new LinkedList<ArrayList<Particle>>();
    weights = new double[numParticles];
    nwQueue = new LinkedList<ArrayList<int[][]>>();
    nwsumQueue = new LinkedList<ArrayList<int[]>>();
  }

  void initialize(LDAOptions options) {
    this.options = options;
    K = options.topicNum;
    beta = options.beta;
    corpus = new LDACorpus();
    loadCorpus();
    initParticles();
  }

  /**
   * Load training corpus for initialization
   */
  private void loadCorpus() {
    corpus.loadStopwords(options.dir + File.separator + options.sfile);
    corpus.loadDocs(options.dir + File.separator + options.cfile);
    this.V = corpus.vocabulary.V;
  }

  private void initParticles() {
    ArrayList<int[][]> nw = new ArrayList<int[][]>(numParticles);
    ArrayList<int[]> nwsum = new ArrayList<int[]>(numParticles);
    int[][] nw0 = new int[V][K];
    for (int w = 0; w < V; w++) {
      for (int k = 0; k < K; k++) {
        nw0[w][k] = 0;
      }
    }

    for (int p = 0; p < numParticles; p++) {
      nw.add(nw0);
    }
    nwQueue.addFirst(nw);

    int[] nwsum0 = new int[K];
    for (int k = 0; k < K; k++) {
      nwsum0[k] = 0;
    }

    for (int p = 0; p < numParticles; p++) {
      nwsum.add(nwsum0);
    }
    nwsumQueue.addFirst(nwsum);

    int D = corpus.docs.size();
    ArrayList<Particle> particles = new ArrayList<Particle>();
    for (int p = 0; p < numParticles; p++) {
      // initialize alpha
      double[] alpha = new double[K - 1];
      for (int k = 0; k < K - 1; k++) {
        alpha[k] = rand.nextGaussian();
      }

      // initialize eta
      ArrayList<double[]> etaList = new ArrayList<double[]>();
      for (int d = 0; d < D; d++) {
        double[] eta = new double[K];
        for (int k = 0; k < K - 1; k++) {
          eta[k] = alpha[k] + rand.nextGaussian();
        }
        eta[K - 1] = 0.0;
        etaList.add(eta);
      }
      // initialze z
      ArrayList<int[]> zList = new ArrayList<int[]>();
      for (int d = 0; d < D; d++) {
        int N = corpus.docs.get(d).length;
        int[] z = new int[N];
        for (int n = 0; n < N; n++) {
          // Topic assignments are initialized to values in [0, K - 1] for each word.
          int topic = rand.nextInt(K);
          z[n] = topic;
          // number of instances of word i (i = documents[m][n]) assigned to topic j (j=z[m][n])
          nw.get(p)[corpus.docs.get(d).words[n]][topic]++;
          // total number of words assigned to topic j.
          nwsum.get(p)[topic]++;
        }
        zList.add(z);
      }
      particles.add(new Particle(zList, etaList, alpha));
    }
    // add initialized particle to the queue
    this.particlesQueue.addFirst(particles);
  }

  void setObserve(ArrayList<Document> observe) {
    this.observe = observe;
  }

  /**
   * Likelihood function
   *
   * @param particle topic assignments for observe
   * @param nw cumulative topic word counts matrix belonging to the same path of the particle
   * @param nwsum cumulative word counts sum corresponding to nw
   * @return the value of likelihood
   */
  double likelihood(Particle particle, int[][] nw, int[] nwsum) {
    double probability = 1.0;
    int D = observe.size();
    for (int d = 0; d < D; d++) {
      double thetaSum = 0.0;
      for (int k = 0; k < K; k++) {
        thetaSum += Math.exp(particle.eta.get(d)[k]);
      }

      int N = observe.get(d).length;
      for (int n = 0; n < N; n++) {
        int topic = particle.z.get(d)[n];
        int word = observe.get(d).words[n];
        nw[word][topic] -= 1;
        nwsum[topic] -= 1;
        double Vbeta = V * beta;

        double probWz = (nw[word][topic] + beta) / (nwsum[topic] + Vbeta);
        nw[word][topic] += 1;
        nwsum[topic] += 1;
        double probZeta = Math.exp(particle.eta.get(d)[topic]) / thetaSum;
        probability *= (probWz * probZeta);
      }

//      double probEtaAlpha = 1.0;
//      for (int k = 0; k < K - 1; k++) {
//        probEtaAlpha *= gaussian(particle.eta.get(d)[k], particle.alpha[k], 1);
//      }
//      probability *= probEtaAlpha;
    }
    return probability;
  }

  /**
   * Predictive likelihood function
   * For new observation, words have not been assigned to topics
   * when calculating predictive likelihood. Thus, the count of each word doesn't need
   * to be deducted from topic word matrix. Choose the max topic counts from nw as
   * the predictive topic assignment for each word in the new observation.
   * And use mean of thetas from previous particle.
   *
   * @param nw cumulative topic word counts matrix up to previous observations
   * @param nwsum cumulative word counts sum up to previous observations
   * @return the value of predictive likelihood
   */
  double preLikelihood(Particle preParticle, int[][] nw, int[] nwsum) {
    double[] meanTheta = new double[K];
    for (int k = 0; k < K; k++) {
      double thetaSum = 0.0;
      for (double[] eta : preParticle.eta) {
        thetaSum += Math.exp(eta[k]);
      }
      meanTheta[k] = thetaSum;
    }
    double sum = 0.0;
    for (double theta : meanTheta) {
      sum += theta;
    }
    for (int k = 0; k < K; k++) {
      meanTheta[k] /= sum;
    }

    double Vbeta = V * beta;
    double probability = 1.0;
    for (Document doc : observe) {
      int N = doc.length;
      for (int n = 0; n < N; n++) {
        int word = doc.words[n];
        int topic = 0;
        int topicCount = nw[word][0];
        for (int k = 1; k < K; k++) {
          if (topicCount < nw[word][k]) {
            topicCount = nw[word][k];
            topic = k;
          }
        }
        double probWz = (nw[word][topic] + beta) / (nwsum[topic] + Vbeta);
        double probZeta = meanTheta[topic];
        probability *= (probWz * probZeta);
      }
    }
    return probability;
  }

  void updatePreWeights() {
    for (int p = 0; p < numParticles; p++) {
      weights[p] *= preLikelihood(particlesQueue.getFirst().get(p), nwQueue.getFirst().get(p),
          nwsumQueue.getFirst().get(p));
    }
  }

  void resample(double[] weights, int numResamples) {
    int[] resampleCounts = new int[numParticles];
    double ran = rand.nextDouble() / numResamples;
    double weightCumulative = 0.0;
    for (int i = 0; i < numResamples; i++) {
      int count = 0;
      weightCumulative += weights[i];
      while (weightCumulative > ran) {
        count++;
        ran += 1.0 / numResamples;
      }
      resampleCounts[i] = count;
    }
    // map resample count to resample index
    int[] resampleIndex = new int[numParticles];
    for (int i = 0, j = 0; i < numResamples; i++) {
      while (resampleCounts[i] > 0 && j < numResamples) {
        resampleIndex[j] = i;
        resampleCounts[i]--;
        j++;
      }
    }
    resampleIndexQueue.addFirst(resampleIndex);
    if (resampleIndexQueue.size() > windowSize){
      resampleIndexQueue.removeLast();
    }
  }

  void propagate() {
    ArrayList<Particle> preParticles = particlesQueue.getFirst();
    ArrayList<Particle> particles = new ArrayList<Particle>(numParticles);
    ArrayList<int[][]> nws = new ArrayList<int[][]>(numParticles);
    ArrayList<int[]> nwsums = new ArrayList<int[]>(numParticles);
    for (int index : resampleIndexQueue.getFirst()) {
      // propagate alpha
      double[] alpha = new double[K - 1];
      for (int k = 0; k < K - 1; k++) {
        alpha[k] = rand.nextGaussian() + preParticles.get(index).alpha[k];
      }

      int D = observe.size();

      // propagate eta
      ArrayList<double[]> etaList = new ArrayList<double[]>(D);
      for (int d = 0; d < D; d++) {
        double[] eta = new double[K - 1];
        for (int k = 0; k < K - 1; k++) {
          eta[k] = rand.nextGaussian() + alpha[k];
        }
        eta[K] = 0.0;
        etaList.add(eta);
      }

      // propagate z
      ArrayList<int[]> zList = new ArrayList<int[]>(D);

      int[][] nw = new int[V][K];
      int[] nwsum = new int[V];
      for (int k = 0; k < K; k++) {
        for (int w = 0; w < V; w++) {
          nw[w][k] = nwQueue.getFirst().get(index)[w][k];
        }
        nwsum[k] = nwsumQueue.getFirst().get(index)[k];
      }

      double Vbeta = V * beta;
      for (int d = 0; d < D; d++) {
        int N = observe.get(d).length;
        int[] z = new int[N];
        for (int n = 0; n < N; n++) {
          int word = observe.get(d).words[n];
          double[] prob = new double[K];
          for (int k = 0; k < K; k++) {
            prob[k] = (nw[word][k] + beta) / (nwsum[k] + Vbeta) * Math.exp(etaList.get(d)[k]);
          }
          // Compute cumulative probability
          for (int k = 1; k < K; k++) {
            prob[k] += prob[k - 1];
          }
          // scaled sample because of unnormalized p[]
          double u = rand.nextDouble() * prob[K - 1];

          int topic;
          for (topic = 0; topic < K; topic++) {
            if (prob[topic] > u)
              break;
          }
          // Add newly estimated topic assignment z to count variables.
          nw[word][topic] += 1;
          nwsum[topic] += 1;
          z[n] = topic;
        }
        zList.add(z);
      }
      particles.add(new Particle(zList, etaList, alpha));
      nws.add(nw);
      nwsums.add(nwsum);
    }
    particlesQueue.addFirst(particles);
    if (particlesQueue.size() > windowSize) {
      particlesQueue.removeLast();
    }
    nwQueue.addFirst(nws);
    if (nwQueue.size() > windowSize) {
      nwQueue.removeLast();
    }
    nwsumQueue.addFirst(nwsums);
    if (nwsumQueue.size() > windowSize) {
      nwsumQueue.removeLast();
    }
  }

  void updateWeights() {
    for (int p = 0; p < numParticles; p++) {
      int[] resampleIndex = resampleIndexQueue.getFirst();
      weights[p] = likelihood(particlesQueue.getFirst().get(p), nwQueue.getFirst().get(p),
          nwsumQueue.getFirst().get(p)) / preLikelihood(particlesQueue.get(1).get(resampleIndex[p]),
          nwQueue.get(1).get(resampleIndex[p]), nwsumQueue.get(1).get(resampleIndex[p]));
    }
    double weightSum = 0.0;
    for (double weight : weights) {
      weightSum += weight;
    }
    for (int p = 0; p < numParticles; p++) {
      weights[p] /= weightSum;
    }
  }

  private double gaussian(double x, double mu , double sigma) {
    return Math.exp(-Math.pow(mu - x, 2) / Math.pow(sigma, 2) / 2.0)
        / Math.sqrt(2.0 * Math.PI * Math.pow(sigma, 2));
  }
}


