package ding.del.lda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LDA {

  public static void main(String[] args) {
    LDAOptions options = new LDAOptions(args);
//    GibbsSampler sampler = new GibbsSampler();
//    sampler.initialize(options);
//    sampler.estimate();
//    sampler.trnModel.corpus.saveVocabulary(options.dir + File.separator + options.vfile);

    Vocabulary nipsVocab = new Vocabulary();
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(options.dir + File.separator + options.vfile), "UTF-8"));
      String line;
      int index = 1;
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split("\\s");
        String word = tokens[0];
        nipsVocab.indexToWord.put(index, word);
        nipsVocab.wordToIndex.put(word, index);
        index++;
      }
      nipsVocab.V = nipsVocab.indexToWord.size();
      reader.close();
    } catch (Exception e) {
      System.err.println("Error while reading vocabulary: " + e.getMessage());
    }

    LDACorpus nips = new LDACorpus(1500); // 1500 docs in NIPS corpus
    nips.vocabulary = nipsVocab;

    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(options.dir + File.separator + options.cfile), "UTF-8"));
      reader.readLine(); // number of docs: D
      reader.readLine(); // number of words: V
      reader.readLine(); // number of non-zero counts: NNZ

      String line;
      int docNum = 1;
      ArrayList<Integer> doc = new ArrayList<Integer>();
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split("\\s");
        int docId = Integer.parseInt(tokens[0]);
        Integer wordId = Integer.parseInt(tokens[1]);
        int wordCount = Integer.parseInt(tokens[2]);
        if (docId == docNum) {
          while (wordCount > 0) {
            doc.add(wordId);
            wordCount--;
          }
        } else {
          nips.docs.add(new Document(doc));
          doc = new ArrayList<Integer>();
          docNum = docId;
          while (wordCount > 0) {
            doc.add(wordId);
            wordCount--;
          }
        }
      }
      reader.close();
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }

    ParticleFilter pf = new ParticleFilter(5, 500);
    pf.options = options;
    pf.K = options.topicNum;
    pf.beta = options.beta;
    pf.corpus = nips;
    pf.V = nips.vocabulary.V;
    pf.initParticles();
    for (int i = 0; i < 300; i++) { // 300 iterations, 5 docs per iteration
      pf.run(5); // 5 docs per time slice
    }
  }
}
