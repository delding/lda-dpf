package ding.del.lda;

import java.io.File;

public class LDA {

    public static void main(String[] args) {
        LDAOptions options = new LDAOptions(args);
        GibbsSampler sampler = new GibbsSampler();
        sampler.initialize(options);
        sampler.estimate();
        sampler.trnModel.corpus.saveVocabulary(options.dir + File.separator + options.vfile);
    }

}
