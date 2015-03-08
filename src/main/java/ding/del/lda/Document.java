package ding.del.lda;

import java.util.ArrayList;

public class Document {
  String docName;
  int[] words;
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

  public Document(int length, int[] words) {
    this.length = length;
    docName = "";
    this.words = new int[length];
    System.arraycopy(words, 0, this.words, 0, length);
  }

  public Document(ArrayList<Integer> words) {
    length = words.size();
    docName = "";
    this.words = new int[length];
    for (int i = 0; i < length; ++i) {
      this.words[i] = words.get(i);
    }
  }

  public Document(int length, int[] words, String docName) {
    this.length = length;
    this.docName = docName;
    this.words = new int[length];
    System.arraycopy(words, 0, this.words, 0, length);
  }
}