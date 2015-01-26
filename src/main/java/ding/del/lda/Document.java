package ding.del.lda;

public class Document {
    String docName;
    int [] words;
    int length;

    public Document() {
        words = null;
        docName = "";
        length = 0;
    }

    public Document(int length) {
        this.length = length;
        docName = "";
        words = new int[length];
    }

    public Document(int length, int [] words) {
        this.length = length;
        docName = "";
        this.words = new int[length];
        System.arraycopy(words, 0, this.words, 0, length);
    }

    public Document(int length, int [] words, String docName) {
        this.length = length;
        this.docName = docName;
        this.words = new int[length];
        System.arraycopy(words, 0, this.words, 0, length);
    }
}
