package ding.del.lda;

import java.io.File;

public class Gibbs {
    LDAModel trnModel; //model to be trained
    LDAOptions options;

    public void initialize(LDAOptions options) {
        this.options = options;
        trnModel = new LDAModel();
        trnModel.initialize(options);
        trnModel.data.vocabulary.writeVocabulary(options.dir + File.separator + options.vocabularyFile);
    }

    public void estimate() {
        System.out.println("Sampling " + trnModel.iterationNum + " iterations!");
        for (int i = 0; i < trnModel.iterationNum; ++i) {
            System.out.println("Iteration " + i + " ...");

            if ((i >= trnModel.saveStep) && ((i - trnModel.saveIter) % options.saveStep == 0)) {
                System.out.println("Saving the model at iteration " + i + " ...");
                computeTheta();
                computePhi();
                trnModel.saveModel("model-iteration-" + trnModel.saveIter);
            }

            for (int m = 0; m < trnModel.M; m++) {
                for (int n = 0; n < trnModel.data.docs.get(m).length; n++) {
                    int newTopic = sampling(m, n);
                    trnModel.z[m][n] = newTopic;
                }
            }
        }

        System.out.println("Gibbs sampling completed!\n");
        System.out.println("Saving the final model!\n");
        computeTheta();
        computePhi();
        trnModel.saveModel("model-final");
    }

    /**
     * Do sampling
     * @param m document number
     * @param n word number
     * @return topic assignment index
     */
    public int sampling(int m, int n){
        // remove z_i from the count variable
        int topic = trnModel.z[m][n];
        int w = trnModel.data.docs.get(m).words[n];
        trnModel.nw[w][topic] -= 1;
        trnModel.nd[m][topic] -= 1;
        trnModel.nwsum[topic] -= 1;
        trnModel.ndsum[m] -= 1;

        double Vbeta = trnModel.V * trnModel.beta;
        double Kalpha = trnModel.K * trnModel.alpha;

        // Compute p(z_i = k | z_-i, w) via collapsed Gibbs sampling
        double [] p = new double[trnModel.K];
        for (int k = 0; k < trnModel.K; k++) {
            p[k] = (trnModel.nw[w][k] + trnModel.beta)/(trnModel.nwsum[k] + Vbeta) *
                    (trnModel.nd[m][k] + trnModel.alpha)/(trnModel.ndsum[m] + Kalpha);
        }

        // Sample a new topic label for w_{m, n} from multinomial (p[0], p[1], ... , p[K-1])
        // Compute cumulative probability for p
        for (int k = 1; k < trnModel.K; k++){
            p[k] += p[k - 1];
        }

        // scaled sample because of unnormalized p[]
        double u = Math.random() * p[trnModel.K - 1];

        for (topic = 0; topic < trnModel.K; topic++) {
            if (p[topic] > u)
                break;
        }

        // add newly estimated topic assignment z_i to count variables
        trnModel.nw[w][topic] += 1;
        trnModel.nd[m][topic] += 1;
        trnModel.nwsum[topic] += 1;
        trnModel.ndsum[m] += 1;

        return topic;
    }

    public void computeTheta(){
        for (int m = 0; m < trnModel.M; m++) {
            for (int k = 0; k < trnModel.K; k++) {
                trnModel.theta[m][k] = (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
            }
        }
    }

    public void computePhi(){
        for (int k = 0; k < trnModel.K; k++){
            for (int w = 0; w < trnModel.V; w++){
                trnModel.phi[k][w] = (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
            }
        }
    }
}