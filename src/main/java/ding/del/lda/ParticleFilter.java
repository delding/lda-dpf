package ding.del.lda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Particle filter for inference of streaming documents
 */
public class ParticleFilter {
  LDACorpus corpus =  null; // a single observation in this corpus contains a mini-batch of documents
  LDAOptions options;

  int V;
  int K = 100;
  double beta = 0.1;
  // standard deviation of alpha and eta
  // A bigger std can make document-topic distribution more concentrated on a few topics.
  // This produces a similar effect as choosing small hyper-parameter for dirichlet prior.
  double alphaStd = 0.5;
  double etaStd = 0.2;

  int windowSize; // length of sliding window
  int numParticles;

  // Each item in the list is an array of Particles, the length of array is numParticle.
  // This array contains particles given the observation of the documents in one time slice
  LinkedList<ArrayList<Particle>> particlesQueue;

  // Each array in the queue is the resample index for particles in that time slice
  LinkedList<int[]> resampleIndexQueue;

  // store log weight of each particle path ,length = numParticles
  double[] logWeights;

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
    logWeights = new double[numParticles];
    nwQueue = new LinkedList<ArrayList<int[][]>>();
    nwsumQueue = new LinkedList<ArrayList<int[]>>();
  }

  public void setOptions(LDAOptions options) {
    this.options = options;
    K = options.topicNum;
    beta = options.beta;
  }

  public void setCorpus(LDACorpus corpus) {
    this.corpus = corpus;
    this.V = corpus.vocabulary.V;
  }

  /**
   * Initialize particle filter
   *
   * @param numDocs number of documents used for initialization
   */
  public void initialize(int numDocs) {
    setObserve(numDocs);
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

    int[] nwsum0 = new int[K];
    for (int k = 0; k < K; k++) {
      nwsum0[k] = 0;
    }
    for (int p = 0; p < numParticles; p++) {
      nwsum.add(nwsum0);
    }

    ArrayList<Particle> particles = new ArrayList<Particle>();
    for (int p = 0; p < numParticles; p++) {
      // initialze alpha
      double[] alpha = new double[K - 1];
      for (int k = 0; k < K - 1; k++) {
        alpha[k] = rand.nextGaussian() * alphaStd;
      }
      // initialze eta
      ArrayList<double[]> etaList = new ArrayList<double[]>();
      for (int d = 0; d < numDocs; d++) {
        double[] eta = new double[K];
        for (int k = 0; k < K - 1; k++) {
          eta[k] = alpha[k] + rand.nextGaussian() * etaStd;
        }
        eta[K - 1] = 0.0;
        etaList.add(eta);
      }
      // initialize z
      ArrayList<int[]> zList = new ArrayList<int[]>();
      for (int d = 0; d < numDocs; d++) {
        int N = observe.get(d).length;
        int[] z = new int[N];
        for (int n = 0; n < N; n++) {
          // Topic assignments are initialized to values in [0, K - 1] for each word.
          int topic = rand.nextInt(K);
          z[n] = topic;
          // number of instances of word i (i = documents[m][n]) assigned to topic j (j=z[m][n])
          nw.get(p)[observe.get(d).words[n]][topic]++;
          // total number of words assigned to topic j.
          nwsum.get(p)[topic]++;
        }
        zList.add(z);
      }
      particles.add(new Particle(zList, etaList, alpha));
    }
    // initialize log weights
    for (int p = 0; p < numParticles; p++) {
      logWeights[p] = 0.0;
    }
    // add initialized particle to the queue
    particlesQueue.addFirst(particles);
    nwQueue.addFirst(nw);
    nwsumQueue.addFirst(nwsum);
  }

  double[] getNormalizedWeights() {
    double[] weights = new double[numParticles];
    double max = logWeights[0];
    for (int i = 1; i < numParticles; i++) {
      if (max < logWeights[i]) {
        max = logWeights[i];
      }
    }
    double sum = 0.0;
    // Scale log weights by subtracting max weight from each component.
    for (int i = 0; i < numParticles; i++) {
      logWeights[i] -= max;
      weights[i] = Math.exp(logWeights[i]);
      sum += weights[i];
    }
    for (int i = 0; i < numParticles; i++) {
      weights[i] /= sum;
    }
    return weights;
  }

  /**
   * Set documents number for each mini-batch
   *
   * @param numDocsPerOb number of documents per observation
   */
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
   * @return the value of logLikelihood
   */
  double logLikelihood(Particle particle, int[][] nw, int[] nwsum) {
    double logProb = 0.0;
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

        // log probability of w given z
        double logProbWz = Math.log((nw[word][topic] + beta) / (nwsum[topic] + Vbeta));
        nw[word][topic] += 1;
        nwsum[topic] += 1;
        // log probability of z given eta
        double logProbZeta = Math.log(Math.exp(particle.eta.get(d)[topic]) / thetaSum);
        logProb += (logProbWz + logProbZeta);
      }
      // double probEtaAlpha = 1.0;
      //  for (int k = 0; k < K - 1; k++) {
      //  probEtaAlpha *= gaussian(particle.eta.get(d)[k], particle.alpha[k], 1);
      //  }
      // probability *= probEtaAlpha;
    }
    return logProb;
  }

  /**
   * Predictive log likelihood function
   * For new observation, words have not been assigned to topics
   * when calculating predictive log likelihood. Thus, the count of each word doesn't need
   * to be deducted from topic word matrix. Choose the max topic counts from nw as
   * the predictive topic assignment for each word in the new observation.
   * And use mean of thetas from previous particle.
   *
   * @param nw cumulative topic word counts matrix up to previous observations
   * @param nwsum cumulative word counts sum up to previous observations
   * @return the value of predictive logLikelihood
   */
  double preLogLikelihood(Particle preParticle, int[][] nw, int[] nwsum) {
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
    double logProb = 0.0;
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
        double logProbWz = Math.log((nw[word][topic] + beta) / (nwsum[topic] + Vbeta));
        double logProbZeta = Math.log(meanTheta[topic]);
        logProb += (logProbWz + logProbZeta);
      }
    }
    return logProb;
  }

  void updatePreLogWeights() {
    for (int p = 0; p < numParticles; p++) {
      logWeights[p] += preLogLikelihood(particlesQueue.getFirst().get(p), nwQueue.getFirst().get(p),
          nwsumQueue.getFirst().get(p));
    }
  }

  void resample() {
    resample(numParticles);
  }

  /**
   * Systematic resampling
   *
   * @param numResamples number of resampled particles
   */
  void resample(int numResamples) {
    double[] weights = getNormalizedWeights();
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
    if (resampleIndexQueue.size() > windowSize) {
      resampleIndexQueue.removeLast();
    }
  }

  /**
   * Propagate particles to next time slice
   */
  void propagate() {
    ArrayList<Particle> preParticles = particlesQueue.getFirst();
    ArrayList<Particle> particles = new ArrayList<Particle>(numParticles);
    ArrayList<int[][]> nws = new ArrayList<int[][]>(numParticles);
    ArrayList<int[]> nwsums = new ArrayList<int[]>(numParticles);
    ArrayList<double[]> etaList;
    ArrayList<int[]> zList;
    for (int index : resampleIndexQueue.getFirst()) {
      // propagate alpha
      double[] alpha = new double[K - 1];
      for (int k = 0; k < K - 1; k++) {
        alpha[k] = rand.nextGaussian() * alphaStd  + preParticles.get(index).alpha[k];
      }

      int D = observe.size();

      // propagate eta
      etaList = new ArrayList<double[]>(D);
      for (int d = 0; d < D; d++) {
        double[] eta = new double[K];
        for (int k = 0; k < K - 1; k++) {
          eta[k] = rand.nextGaussian() * etaStd + alpha[k];
        }
        eta[K - 1] = 0.0;
        etaList.add(eta);
      }

      // propagate z
      zList = new ArrayList<int[]>(D);
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
      // TODO: remove count contributions too
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

  void updateLogWeights() {
    int[] resampleIndex = resampleIndexQueue.getFirst();
    for (int p = 0; p < numParticles; p++) {
      // window size is at least equal to 2
      logWeights[p] = logLikelihood(particlesQueue.getFirst().get(p), nwQueue.getFirst().get(p),
          nwsumQueue.getFirst().get(p)) - preLogLikelihood(particlesQueue.get(1).get(resampleIndex[p]),
          nwQueue.get(1).get(resampleIndex[p]), nwsumQueue.get(1).get(resampleIndex[p]));
    }
  }

  public void run(int numDocsPerOb) {
    setObserve(numDocsPerOb);
    updatePreLogWeights();
    resample();
    propagate();
    updateLogWeights();
  }

  /**
   * @param topParticleNum
   * @return most weighted particles
   */
  public int[] topWeightIdx(int topParticleNum) {
    int[] topIdx = new int[topParticleNum];
    Map<Double, Integer> sortedMap = new TreeMap<Double, Integer>();
    for (int p = 0; p < numParticles; p++) {
      // multiplying -1 makes max log weight become first value in natural ordering
      sortedMap.put(-1 * logWeights[p], p);
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
    double[] weights = getNormalizedWeights();
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
    double[] weights = getNormalizedWeights();
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
    double[] weights = getNormalizedWeights();
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
    double[] weights = getNormalizedWeights();
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
            WordFreqPair pair = new WordFreqPair(w, phi[k][w]);
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

  /**
   * compute perplexity on hold out (test) data set
   *
   * @param docs incoming or test documents set
   */
  public double computePerplexity(ArrayList<Document> docs) {
    observe = docs;
    updatePreLogWeights();
    resample();
    propagate();
    updateLogWeights();

    double[] weights = getNormalizedWeights();
    double Vbeta = V * beta;
    double perplexity = 0.0;
    for (int p = 0; p < numParticles; p++) {
      int numWords = 0;
      double logProb = 0.0;
      for (int d = 0; d < docs.size(); d++) {
        double[] eta = particlesQueue.getFirst().get(p).eta.get(d);
        double thetaSum = 0.0;
        for (double anEta : eta) {
          thetaSum += Math.exp(anEta);
        }
        double[] theta = new double[K];
        for (int k = 0; k < K; k++) {
          theta[k] = Math.exp(theta[k]) / thetaSum;
        }
        for (int w : docs.get(d).words) {
          numWords++;
          double probW = 0.0;
          // sume over all topics
          for (int k = 0; k < K; k++) {
            probW += (nwQueue.getFirst().get(p)[w][k] + beta)
                / (nwsumQueue.getFirst().get(p)[k] + Vbeta) * theta[k];
          }
          logProb += Math.log(probW);
        }
      }
      perplexity += Math.exp(-1 * logProb / numWords) * weights[p];
    }
    return perplexity;
  }

  /**
   * initialize from existing file
   *
   * @param nwFile file that stores word-topic counts
   */
  public void initFromFile(String nwFile) {
    int[][] nw0 = new int[V][K];
    int[] nwsum0 = new int[K];
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(nwFile), "UTF-8"));
      String line;
      int w = 0;
      while ((line = reader.readLine()) != null) {
        String[] counts = line.split("\\s");
        for (int k = 0; k < K; k++) {
          nw0[w][k] = Integer.parseInt(counts[k]);
        }
        w++;
      }
      reader.close();
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }
    for (int k = 0; k < K; k++) {
      nwsum0[k] = 0;
      for (int w = 0; w < V; w++) {
        nwsum0[k] += nw0[w][k];
      }
    }
    // initialize alpha
    double[] alpha = new double[K - 1];
    for (int k = 0; k < K - 1; k++) {
      alpha[k] = rand.nextGaussian() * alphaStd;
    }
    // initialze eta, set d = 1
    ArrayList<double[]> etaList = new ArrayList<double[]>();
    for (int d = 0; d < 1; d++) {
      double[] eta = new double[K];
      for (int k = 0; k < K - 1; k++) {
        eta[k] = alpha[k] + rand.nextGaussian() * etaStd;
      }
      eta[K - 1] = 0.0;
      etaList.add(eta);
    }

    ArrayList<int[][]> nw = new ArrayList<int[][]>();
    ArrayList<int[]> nwsum = new ArrayList<int[]>();
    ArrayList<Particle> particles = new ArrayList<Particle>();
    for (int p = 0; p < numParticles; p++) {
      nw.add(nw0);
      nwsum.add(nwsum0);
      particles.add(new Particle(alpha, etaList));
    }
    // initialize log weights
    for (int p = 0; p < numParticles; p++) {
      logWeights[p] = 0.0;
    }
    // add initialized particle to the queue
    particlesQueue.addFirst(particles);
    nwQueue.addFirst(nw);
    nwsumQueue.addFirst(nwsum);
  }

  private double gaussian(double x, double mu , double sigma) {
    return Math.exp(-Math.pow(mu - x, 2) / Math.pow(sigma, 2) / 2.0)
        / Math.sqrt(2.0 * Math.PI * Math.pow(sigma, 2));
  }
}


