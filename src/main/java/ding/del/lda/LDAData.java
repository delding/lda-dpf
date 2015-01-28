package ding.del.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LDAData {

    Vocabulary vocabulary;
    Stopwords stopwords;
    ArrayList<Document> docs;

    public LDAData() {
        vocabulary = new Vocabulary();
        stopwords = new Stopwords();
        docs = new ArrayList<Document>();
    }

    public LDAData(int M) {
        vocabulary = new Vocabulary();
        stopwords = new Stopwords();
        docs = new ArrayList<Document>(M);
    }

    public LDAData(int M, String sfile) {
        vocabulary = new Vocabulary();
        stopwords = new Stopwords();
        loadStopwords(sfile);
        docs = new ArrayList<Document>(M);
    }

    public LDAData(String sfile) {
        vocabulary = new Vocabulary();
        stopwords = new Stopwords();
        loadStopwords(sfile);
        docs = new ArrayList<Document>();
    }

    public void loadStopwords(String filename) {
        stopwords.loadStopwords(filename);
    }

    public void saveVocabulary(String filename) {
        vocabulary.writeVocabulary(filename);
    }

    /**
     * @param filename
     *  read documents from a file, each line contains a document, create new vocabulary
     */
    public void loadDocs(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(filename), "UTF-8"));
            loadDocs(reader);
            reader.close();
        }
        catch (Exception e) {
            System.out.println("Read Data Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     *  read a data from a stream, each line contains a document, create new vocabulary
     */
    public void loadDocs(BufferedReader reader) {
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                addDoc(line);
            }
        }
        catch (Exception e){
            System.out.println("Read Data Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * read a data from a string, create new dictionary
     * @param strs String from which we get the data, documents are separated by newline character
     */
    public void loadDocs(String[] strs) {
        for (String str : strs) {
            addDoc(str);
        }
    }

    /**
     * add the document to docs
     * stop words are filtered, vocabulary is updated
     * @param doc document to be set
     */
    public void addDoc(Document doc) {
        docs.add(doc);
    }

    /**
     * add the document to docs
     * stop words are filtered, vocabulary is updated
     * @param str string contains doc
     */
    public void addDoc(String str) {
        addDoc(createDoc(str));
    }

    /**
     * create a new document from a string
     * stop words are filtered, vocabulary is updated
     * @param str string that contains the document
     * @return new document
     */
    private Document createDoc(String str) {
        String [] words = str.split("\\s");
        ArrayList<Integer> indexes = new ArrayList<Integer>();
        for(String word : words) {
            if (!stopwords.isStopword(word)) {
                if(vocabulary.contains(word)) {
                    indexes.add(vocabulary.getIndex(word));
                } else {
                    indexes.add(vocabulary.addWord(word));
                }
            }
        }
        return new Document(indexes);
    }

    public static class Document {
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

        public Document(ArrayList<Integer> words) {
            length = words.size();
            docName = "";
            this.words = new int[length];
            for (int i = 0; i < length; ++i) {
                this.words[i] = words.get(i);
            }
        }

        public Document(int length, int [] words, String docName) {
            this.length = length;
            this.docName = docName;
            this.words = new int[length];
            System.arraycopy(words, 0, this.words, 0, length);
        }
    }

}
