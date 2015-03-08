package ding.del.lda;

import java.io.*;
import java.util.*;

public class ParticleFilter {
  LDACorpus corpus =  null; // a single observation in this corpus contains a mini-batch of documents
  LDAOptions options;

  int V;
  int K = 50;
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
  int offset = 0; // offset of current observation in corpus

  Random rand = new Random();

  public ParticleFilter(int windowSize, int numParticles) {
    this.windowSize = windowSize;
    this.numParticles = numParticles;
    particlesQueue = new LinkedList<ArrayList<Particle>>();
    resampleIndexQueue = new LinkedList<int[]>();
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

  public void initParticles() {
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
    // initialize weights
    for (int p = 0; p < numParticles; p++) {
      weights[p] = 1.0 / numParticles;
    }
  }

  void setObserve(int numDocsPerOb) {
    ArrayList<Document> observe = new ArrayList<Document>(numDocsPerOb);
    for (int i = 0; i < numDocsPerOb; i++) {
      observe.add(corpus.docs.get(offset + i));
    }
    this.observe = observe;
    offset += numDocsPerOb;
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

  void resample(int numResamples) {
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

  public void run(int numDocsPerOb) {
    setObserve(numDocsPerOb);
    updatePreWeights();
    resample(numParticles);
    propagate();
    updateWeights();
  }

  private double gaussian(double x, double mu , double sigma) {
    return Math.exp(-Math.pow(mu - x, 2) / Math.pow(sigma, 2) / 2.0)
        / Math.sqrt(2.0 * Math.PI * Math.pow(sigma, 2));
  }

  public int[] topWeightIdx(int topParticleNum) {
    int[] topIdx = new int[topParticleNum];
    Map<Double, Integer> sortedMap = new TreeMap<Double, Integer>();
    for (int p = 0; p < numParticles; p++) {
      // multiplying -1 makes max weight become first value in natural ordering
      sortedMap.put(-1 * weights[p], p);
    }
    int i = 0;
    for(Map.Entry<Double, Integer> entry : sortedMap.entrySet()) {
      if ( i == topParticleNum ) {
        break;
      }
      topIdx[i] = entry.getValue();
      i++;
    }
    return topIdx;
  }

  public double[] computeTheta(double[] eta) {
    double[] theta = new double[eta.length];
    double sum = 0.0;
    for (double anEta : eta) {
      sum += Math.exp(anEta);
    }
    for (int k = 0; k < eta.length; k++) {
      theta[k] = Math.exp(eta[k]) / sum;
    }
    return theta;
  }

  public double[][] computePhi(int[][] nw, int[] nwsum) {
    double[][] phi = new double[K][V];
    for (int k = 0; k < K; k++) {
      for (int w = 0; w < V; w++) {
        phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
      }
    }
    return phi;
  }

  public void saveTopicAssign(String filename, int topParticleNum) {
    try {
      int[] topIdx = topWeightIdx(topParticleNum);
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      // Write docs with topic assignments for words and corresponding weights.
      for (int d = 0; d < observe.size(); ++d) {
        for (int n = 0; n < observe.get(d).length; ++n) {
          writer.write(observe.get(d).words[n] + ": ");
          for (int p = 0; p < topParticleNum; p++) {
            writer.write(particlesQueue.getFirst().get(topIdx[p]).z.get(d)[n]
                + " " + weights[topIdx[p]] + ", ");
          }
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  public void saveTheta(String filename, int topParticleNum) {
    try {
      int[] topIdx = topWeightIdx(topParticleNum);
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int d = 0; d < observe.size(); ++d) {
        for (int p = 0; p < topParticleNum; p++) {
          double[] theta = computeTheta(particlesQueue.getFirst().get(topIdx[p]).eta.get(d));
          for (int k = 0; k < K; ++k) {
            writer.write(theta[k] + " ");
          }
          writer.write(": " + weights[topIdx[p]] + "\n");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  public void savePhi(String filename, int topParticleNum) {
    try {
      int[] topIdx = topWeightIdx(topParticleNum);
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int p = 0; p < topParticleNum; p++) {
        double[][] phi = computePhi(nwQueue.getFirst().get(topIdx[p]),
            nwsumQueue.getFirst().get(topIdx[p]));
        writer.write(weights[topIdx[p]] + "\n");
        for (int k = 0; k < K; ++k) {
          for (int w = 0; w < V; ++w) {
            writer.write(phi[k][w] + " ");
          }
          writer.write("\n");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  public void saveTopWords(String filename, int topParticleNum) {
    try {
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
          new FileOutputStream(filename), "UTF-8"));
      int[] topIdx = topWeightIdx(topParticleNum);
      for (int p = 0; p < topParticleNum; p++) {
        writer.write("weight: " + weights[topIdx[p]] + "\n");
        double[][] phi = computePhi(nwQueue.getFirst().get(topIdx[p]),
            nwsumQueue.getFirst().get(topIdx[p]));
        for (int k = 0; k < K; k++) {
          List<WordFreqPair> wordsFreqList = new ArrayList<WordFreqPair>();
          for (int w = 0; w < V; w++) {
            WordFreqPair pair = new WordFreqPair(w + 1, phi[k][w]); // wordId starts from 1
            wordsFreqList.add(pair);
          }
          writer.write("Topic " + k + ":\n");
          Collections.sort(wordsFreqList);
          for (int i = 0; i < options.topWords; i++) {
            String word = corpus.vocabulary.getWord((Integer) wordsFreqList.get(i).first);
            writer.write("\t" + word + " " + wordsFreqList.get(i).second + "\n");
          }
        }
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  public void save(String modelName, int topParticleNum) {
    saveTopicAssign(options.dir + File.separator + modelName + "TopicAssign", topParticleNum);
    saveTheta(options.dir + File.separator + modelName + "Theta", topParticleNum);
    savePhi(options.dir + File.separator + modelName + "Phi", topParticleNum);
    saveTopWords(options.dir + File.separator + modelName + "TopWords", topParticleNum);
  }
}


