package ding.del.lda;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LDAModel {
  String modelName;
  LDAOptions options;
  LDACorpus corpus;

  int D; // number of documents
  int V; // vocabulary size
  int K; // number of topics
  double alpha; // symmetric hyperparameter of document-topic Dirichlet prior
  double beta; // symmetric hyperparameter of topic-word (term) Dirichlet prior


  double[][] theta; // document-topic distributions, size D * K
  double[][] phi; // topic-word distributions, size K * V

  int[][] z; // topic assignments for each word, size D * doc.size()
  int[][] nw; // nw[w][k] denotes number of word/term w assigned to topic k, size V * K
  int[][] nd; // nd[d][k] denotes number of words in document d assigned to topic k, size D * K
  int[] nwsum; // nwsum[k] denotes total number of words assigned to topic k, size K
  int[] ndsum; // ndsum[d] denotes total number of words in document d, size D

  public LDAModel() {
    modelName = "";
    options = null;
    corpus = null;

    D = 0;
    V = 0;
    K = 100;
    alpha = 50.0 / K;
    beta = 0.1;

    theta = null;
    phi = null;
    z = null;

    nw = null;
    nd = null;
    nwsum = null;
    ndsum = null;
  }

  protected void setOptions(LDAOptions options) {
    this.options = options;

    modelName = options.modelName;
    K = options.topicNum;
    alpha = options.alpha;
    beta = options.beta;
  }

  public void setCorpus(LDACorpus corpus) {
    this.corpus = corpus;
    D = corpus.docs.size();
    V = corpus.vocabulary.V;
  }

  public void initialize() {
    nw = new int[V][K]; // default int values in array are zeros
    nd = new int[D][K];
    nwsum = new int[K];
    ndsum = new int[D];

    // The z_i are initialized to values in [0, K - 1] for each word.
    z = new int[D][];
    for (int d = 0; d < D; d++) {
      int N = corpus.docs.get(d).length;
      z[d] = new int[N];
      for (int n = 0; n < N; n++) {
        int topic = (int) (Math.random() * K);
        z[d][n] = topic;
        // number of instances of word i (i = documents[m][n]) assigned to topic j (j=z[m][n])
        nw[corpus.docs.get(d).words[n]][topic]++;
        // number of words in document i assigned to topic j
        nd[d][topic]++;
        // total number of words assigned to topic j.
        nwsum[topic]++;
      }
      // total number of words in document i
      ndsum[d] = N;
    }

    theta = new double[D][K];
    phi = new double[K][V];
  }

  public void initFromFile(String nwFile) {
    nw = new int[V][K];
    nwsum = new int[K];
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(nwFile), "UTF-8"));
      String line;
      int w = 0;
      while ((line = reader.readLine()) != null) {
        String[] counts = line.split("\\s");
        for (int k = 0; k < K; k++) {
          nw[w][k] = Integer.parseInt(counts[k]);
        }
        w++;
      }
      reader.close();
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }
    for (int k = 0; k < K; k++) {
      nwsum[k] = 0;
      for (int w = 0; w < V; w++) {
        nwsum[k] += nw[w][k];
      }
    }
  }

  /**
   * Save word-topic assignments
   */
  public void saveTopicAssign(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      // Write docs with topic assignments for words.
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < corpus.docs.get(i).length; ++j) {
          writer.write(corpus.docs.get(i).words[j] + ":" + z[i][j] + " ");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save theta (document topic distribution)
   */
  public void saveTheta(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < K; ++j) {
          writer.write(theta[i][j] + " ");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save nd
   */
  public void saveNd(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int d = 0; d < D; ++d) {
        for (int k = 0; k < K; ++k) {
          writer.write(nd[d][k] + " ");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save word-topic distribution
   */
  public void savePhi(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < V; ++j) {
          writer.write(phi[i][j] + " ");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * save nw
   */
  public void saveNw(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      for (int w = 0; w < V; ++w) {
        for (int k = 0; k < K; ++k) {
          writer.write(nw[w][k] + " ");
        }
        writer.write("\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save other information of this model
   */
  public void saveParams(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
      writer.write("alpha=" + alpha + "\n");
      writer.write("beta=" + beta + "\n");
      writer.write("ntopics=" + K + "\n");
      writer.write("ndocs=" + D + "\n");
      writer.write("nwords=" + V + "\n");
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save the most likely words in each topic
   */
  public void saveTopWords(String filename) {
    try {
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
              new FileOutputStream(filename), "UTF-8"));

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
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while saving topic assignments: " + e.getMessage());
    }
  }

  /**
   * Save All
   */
  public void saveModel(String modelName) {
    saveTopicAssign(options.dir + File.separator + modelName + "TopicAssign");
    saveParams(options.dir + File.separator + modelName + "Other");
    saveTheta(options.dir + File.separator + modelName + "Theta");
    savePhi(options.dir + File.separator + modelName + "Phi");
    saveTopWords(options.dir + File.separator + modelName + "TopWords");
  }

  /**
   * compute perplexity of hold out documents
   *
   * @param docs hold out documents
   * @param iterNum numbers of Gibbs sampling iteration before computing perplexity
   * @return perplexity
   */
  public double computePerplexity(ArrayList<Document> docs, int iterNum) {
    int DNew = docs.size();
    int[][] ndNew = new int[DNew][K];
    int[] ndsumNew = new int[DNew];
    int[][] zNew = new int[DNew][];
    for (int d = 0; d < DNew; d++) {
      int N = docs.get(d).length;
      zNew[d] = new int[N];
      for (int n = 0; n < N; n++) {
        int topic = (int) (Math.random() * K);
        zNew[d][n] = topic;
//        nw[docs.get(d).words[n]][topic]++;
        ndNew[d][topic]++;
//        nwsum[topic]++;
      }
      ndsumNew[d] = N;
    }
    double Vbeta = V * beta;
    double Kalpha = K * alpha;
    for (int i = 0; i < iterNum; i++) {
      for (int d = 0; d < DNew; d++) {
        for (int n = 0; n < docs.get(d).length; n++) {
          int topic = zNew[d][n];
          int w = docs.get(d).words[n];
//          nw[w][topic] -= 1;
          ndNew[d][topic] -= 1;
//          nwsum[topic] -= 1;
          ndsumNew[d] -= 1;

          double[] p = new double[K];
          for (int k = 0; k < K; k++) {
            p[k] = (nw[w][k] + beta) / (nwsum[k] + Vbeta)
                * (ndNew[d][k] + alpha) / (ndsumNew[d] + Kalpha);
          }

          for (int k = 1; k < K; k++) {
            p[k] += p[k - 1];
          }
          double u = Math.random() * p[K - 1];
          // If topic = K - 2 still no break, topic goes to K - 1, loop ends
          for (topic = 0; topic < K - 1; topic++) {
            if (p[topic] > u)
              break;
          }
          zNew[d][n] = topic;
//          nw[w][topic] += 1;
          ndNew[d][topic] += 1;
//          nwsum[topic] += 1;
          ndsumNew[d] += 1;
        }
      }
    }
    double perplexity = 0.0;
    int numWords = 0;
    double logProb = 0.0;
    for (int d = 0; d < docs.size(); d++) { // hold out documents
      for (int w : docs.get(d).words) {
        numWords++;
        double probW = 0.0;
        // sum over all topics
        for (int k = 0; k < K; k++) {
          probW += (nw[w][k] + beta) / (nwsum[k] + Vbeta)
              * (ndNew[d][k] + alpha) / (ndsumNew[d] + Kalpha);
        }
        logProb += Math.log(probW);
      }
    }
    perplexity += Math.exp(-1 * logProb / numWords);
    return perplexity;
  }
}