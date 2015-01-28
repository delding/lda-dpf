package ding.del.lda;

public class LDA {

    public static void main(String[] args) {
        LDAOptions options = new LDAOptions(args);
        GibbsSampler sampler = new GibbsSampler();
        sampler.initialize(options);
        sampler.estimate();
    }

}
