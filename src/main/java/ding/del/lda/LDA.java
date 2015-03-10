package ding.del.lda;

import java.io.File;

public class LDA {

  public static void main(String[] args) {
    LDAOptions options = new LDAOptions(args);

    LDACorpus nips = new LDACorpus("nips");
    String corpusFile = options.dir + File.separator + options.cfile;
    String vocabFile = options.dir + File.separator + options.vfile;
    nips.loadUciData(corpusFile, vocabFile);

    LDAModel model = new LDAModel();
    model.setOptions(options);
    model.setCorpus(nips);
    GibbsSampler sampler = new GibbsSampler(model);
    sampler.initialize();
    sampler.estimate();

    ParticleFilter pf = new ParticleFilter(2, 50); // window size = 2, particle number = 50
    pf.setOptions(options);
    pf.setCorpus(nips);
    pf.initialize(5);
    for (int i = 0; i < 299; i++) { // 299 iterations, 5 docs per iteration, first 5 used to setOptions
      System.out.println("iteration: " + (i + 1));
      pf.run(5); // 5 docs per time slice
      if (i > 198 && (i + 1) % 5 == 0) {
        pf.saveTopicAssign(options.dir + File.separator + "iter" + (i + 1) + ": TopicAssign", 5);
        pf.saveTopWords(options.dir + File.separator + "iter" + (i + 1) + ": TopWords", 5);
      }
    }
  }
}
