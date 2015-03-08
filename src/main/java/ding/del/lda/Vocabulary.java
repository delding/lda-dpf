package ding.del.lda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

public class Vocabulary {
  int V;
  Map<String, Integer> wordToIndex;
  Map<Integer, String> indexToWord;

  public Vocabulary() {
    V = 0;
    wordToIndex = new HashMap<String, Integer>();
    indexToWord = new HashMap<Integer, String>();
  }

  public String getWord(int index) {
    return indexToWord.get(index);
  }

  public Integer getIndex(String word) {
    return wordToIndex.get(word);
  }

  public boolean contains(String word) {
    return wordToIndex.containsKey(word);
  }

  public boolean contains(int index) {
    return indexToWord.containsKey(index);
  }

  public int addWord(String word) {
    if (!contains(word)) {
      int index = V++;
      wordToIndex.put(word, index);
      indexToWord.put(index, word);
      return index;
    } else return getIndex(word);
  }

  /**
   * Read vocabulary from file, each line contains a word and its index.
   */
  public void readVocabulary(String vfile) {
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
              new FileInputStream(vfile), "UTF-8"));
      String line;
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split("\\s");
        String word = tokens[0];
        int index = Integer.parseInt(tokens[1]);
        indexToWord.put(index, word);
        wordToIndex.put(word, index);
      }
      V = indexToWord.size();
      reader.close();
    } catch (Exception e) {
      System.err.println("Error while reading vocabulary: " + e.getMessage());
    }
  }

  public void writeVocabulary(String vfile) {
    try {
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
              new FileOutputStream(vfile), "UTF-8"));

      for (String word : wordToIndex.keySet()) {
        Integer index = wordToIndex.get(word);
        writer.write(word + " " + index + "\n");
      }
      writer.close();
    } catch (Exception e) {
      System.err.println("Error while writing vocabulary: " + e.getMessage());
    }
  }
}