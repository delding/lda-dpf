package ding.del.lda;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LDAModel {
    String modelName;
    LDAOptions options;
    LDAData data;

    int M; //number of documents
    int V; //vocabulary size
    int K; //number of topics
    double alpha; //symmetric hyperparameter of document-topic Dirichlet prior
    double beta; //symmetric hyperparameter of topic-word(term) Dirichlet prior


    double[][] theta; //theta: document - topic distributions, size M * K
    double[][] phi; // phi: topic-word distributions, size K * V

    int[][] z; //topic assignments for each word, size M * doc.size()
    int[][] nw; //nw[i][j]: number of word/term i assigned to topic j, size V * K
    int[][] nd; //nd[i][j]: number of words in document i assigned to topic j, size M * K
    int[] nwsum; //nwsum[j]: total number of words assigned to topic j, size K
    int[] ndsum; //ndsum[i]: total number of words in document i, size M

    public LDAModel() {
        modelName = "";
        options = null;
        data = null;

        M = 0;
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

    protected void initialize(LDAOptions options) {
        this.options = options;

        modelName = options.modelName;
        K = options.topicNum;
        alpha = options.alpha;
        beta = options.beta;

        data = new LDAData();
        loadData();
    }

    /**
     * Load data for estimation
     */
    private void loadData() {
        data.loadStopwords(options.dir + File.separator + options.sfile);
        data.loadDocs(options.dir + File.separator + options.dfile);

        M = data.docs.size();
        V = data.vocabulary.V;

        nw = new int[V][K];
        for (int w = 0; w < V; w++) {
            for (int k = 0; k < K; k++) {
                nw[w][k] = 0;
            }
        }

        nd = new int[M][K];
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                nd[m][k] = 0;
            }
        }

        nwsum = new int[K];
        for (int k = 0; k < K; k++) {
            nwsum[k] = 0;
        }

        ndsum = new int[M];
        for (int m = 0; m < M; m++) {
            ndsum[m] = 0;
        }

        // The z_i are are initialized to values in [0, K - 1] for each word
        z = new int[M][];
        for (int m = 0; m < M; m++) {
            int N = data.docs.get(m).length;
            z[m] = new int[N];
            for (int n = 0; n < N; n++) {
                int topic = (int) (Math.random() * K);
                z[m][n] = topic;
                // number of instances of word i (i = documents[m][n]) assigned to topic j (j=z[m][n])
                nw[data.docs.get(m).words[n]][topic]++;
                // number of words in document i assigned to topic j
                nd[m][topic]++;
                // total number of words assigned to topic j.
                nwsum[topic]++;
            }
            // total number of words in document i
            ndsum[m] = N;
        }

        theta = new double[M][K];
        phi = new double[K][V];
    }

    /**
     * Save word-topic assignments
     */
    public void saveTopicAssign(String filename) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            //write docs with topic assignments for words
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < data.docs.get(i).length; ++j) {
                    writer.write(data.docs.get(i).words[j] + ":" + z[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.err.println("Error while saving topic assignments: " + e.getMessage());
        }
    }

    /**
     * Save theta (topic distribution)
     */
    public void saveTheta(String filename) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < K; ++j) {
                    writer.write(theta[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        }
        catch (Exception e) {
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
        }
        catch (Exception e) {
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
            writer.write("ndocs=" + M + "\n");
            writer.write("nwords=" + V + "\n");
            writer.close();
        }
        catch(Exception e) {
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
                    String word = data.vocabulary.getWord((Integer)wordsFreqList.get(i).first);
                    writer.write("\t" + word + " " + wordsFreqList.get(i).second + "\n");
                }
            }
            writer.close();
        }
        catch(Exception e) {
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
}