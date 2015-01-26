package ding.del.lda;

public class WordFreqPair implements Comparable<WordFreqPair> {
    public Object first;
    public Comparable second;
    public static boolean naturalOrder = false;

    public WordFreqPair(Object k, Comparable v) {
        first = k;
        second = v;
    }

    public WordFreqPair(Object k, Comparable v, boolean naturalOrder){
        first = k;
        second = v;
        WordFreqPair.naturalOrder = naturalOrder;
    }

    public int compareTo(WordFreqPair other) {
        if (naturalOrder)
            return this.second.compareTo(other.second);
        else return -this.second.compareTo(other.second);
    }
}
