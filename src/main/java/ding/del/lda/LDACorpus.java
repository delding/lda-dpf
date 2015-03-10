package ding.del.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LDACorpus {
  String corpusName;
  Vocabulary vocabulary;
  Stopwords stopwords;
  ArrayList<Document> docs;

  public LDACorpus() {
    corpusName = "";
    vocabulary = new Vocabulary();
    stopwords = new Stopwords();
    docs = new ArrayList<Document>();
  }

  public LDACorpus(String corpusName) {
    this.corpusName = corpusName;
    vocabulary = new Vocabulary();
    stopwords = new Stopwords();
    docs = new ArrayList<Document>();
  }

  /**
   * Load UCI machine learning repository Bag of Words Data Set.
   * Data format is described at https://archive.ics.uci.edu/ml/datasets/Bag+of+Words
   *
   * @param corpusFile name of corpus file
   * @param vocabFile  name of vocabulary file
   */
  public void loadUciData(String corpusFile, String vocabFile) {
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(corpusFile), "UTF-8"));
      int D = Integer.parseInt(reader.readLine().trim()); // number of docs
      vocabulary.V = Integer.parseInt(reader.readLine().trim()); // number of words
      reader.readLine(); // skip NNC (number of non-zero counts)
      String line;
      int currentId = 0; // docId starts from 0
      ArrayList<Integer> doc = new ArrayList<Integer>(D);
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split("\\s");
        int docId = Integer.parseInt(tokens[0]) - 1;
        Integer wordId = Integer.parseInt(tokens[1]) - 1; // wordId starts from 0
        int wordCount = Integer.parseInt(tokens[2]);
        if (docId == currentId) {
          while (wordCount > 0) {
            doc.add(wordId);
            wordCount--;
          }
        } else {
          docs.add(new Document(doc));
          doc = new ArrayList<Integer>();
          currentId = docId;
          while (wordCount > 0) {
            doc.add(wordId);
            wordCount--;
          }
        }
      }
      docs.add(new Document(doc)); // add last doc to the corpus
      reader.close();
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream(vocabFile), "UTF-8"));
      String line;
      // WordId starts from 0 unlike UCI file in which wordId starts from 1.
      int wordId = 0;
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.split("\\s");
        String word = tokens[0];
        vocabulary.indexToWord.put(wordId, word);
        vocabulary.wordToIndex.put(word, wordId);
        wordId++;
      }
      vocabulary.V = vocabulary.indexToWord.size();
      reader.close();
    } catch (Exception e) {
      System.err.println("Error while reading vocabulary: " + e.getMessage());
    }
  }

  public void loadStopwords(String filename) {
    stopwords.loadStopwords(filename);
  }

  public void saveVocabulary(String filename) {
    vocabulary.writeVocabulary(filename);
  }

  /**
   * @param filename read documents from a file, each line contains a document, create new vocabulary
   */
  public void loadDocs(String filename) {
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(
              new FileInputStream(filename), "UTF-8"));
      loadDocs(reader);
      reader.close();
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }
  }

  /**
   * read a corpus from a stream, each line contains a document, create new vocabulary
   */
  public void loadDocs(BufferedReader reader) {
    try {
      String line;
      while ((line = reader.readLine()) != null) {
        addDoc(line);
      }
    } catch (Exception e) {
      System.out.println("Read Data Error: " + e.getMessage());
      e.printStackTrace();
    }
  }

  /**
   * Read a corpus from a string, create new dictionary..
   *
   * @param strs strings from which we get the corpus, documents are separated by newline character
   */
  public void loadDocs(String[] strs) {
    for (String str : strs) {
      addDoc(str);
    }
  }

  /**
   * Add the document to docs, stop words are filtered, vocabulary is updated.
   *
   * @param doc document to be set
   */
  public void addDoc(Document doc) {
    docs.add(doc);
  }

  /**
   * Add the document to docs, stop words are filtered, vocabulary is updated
   *
   * @param str string contains doc
   */
  public void addDoc(String str) {
    addDoc(createDoc(str));
  }

  /**
   * Create a new document from a string, stop words are filtered, vocabulary is updated
   *
   * @param str string that contains the document
   * @return new document
   */
  private Document createDoc(String str) {
    String[] words = str.split("\\s");
    ArrayList<Integer> indexes = new ArrayList<Integer>();
    for (String word : words) {
      if (!stopwords.isStopword(word)) {
        if (vocabulary.contains(word)) {
          indexes.add(vocabulary.getIndex(word));
        } else {
          indexes.add(vocabulary.addWord(word));
        }
      }
    }
    return new Document(indexes);
  }
}
