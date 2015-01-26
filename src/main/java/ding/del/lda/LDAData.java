package ding.del.lda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LDAData {

    Vocabulary vocabulary;
    ArrayList<Document> docs;
    int M; 			 		// number of documents
    int V;			 		// number of words in the vocabulary

    public LDAData() {
        vocabulary = new Vocabulary();
        M = 0;
        V = 0;
        docs = new ArrayList<Document>();
    }

    public LDAData(int M) {
        vocabulary = new Vocabulary();
        this.M = M;
        this.V = 0;
        docs = new ArrayList<Document>(M);
    }

    /**
     * set the document at the index idx if idx is greater than 0 and less than M
     * @param doc document to be set
     * @param idx index in the document array
     */
    public void setDoc(Document doc, int idx) {
        if (0 <= idx && idx < M) {
            docs.add(idx, doc);
        }
    }

    /**
     * set the document at the index idx if idx is greater than 0 and less than M
     * @param str string contains doc
     * @param idx index in the document array
     */
    public void setDoc(String str, int idx) {
        if (0 <= idx && idx < M) {
            String [] words = str.split("\\s");
            int length = words.length;
            int [] indexes = new int[length];

            for (int i = 0; i < length; ++i) {
                indexes[i] = vocabulary.getIndex(words[i]);
            }

            Document doc = new Document(length, indexes);
            docs.add(idx, doc);
        }
    }

    /**
     *  read a data from a stream, create new vocabulary
     *  @return LDAData if success and null otherwise
     */
    public static LDAData readData(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(filename), "UTF-8"));
            LDAData data = readData(reader);
            reader.close();
            return data;
        }
        catch (Exception e){
            System.out.println("Read Data Error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     *  read a data from a stream, create new vocabulary
     *  @return LDAData if success and null otherwise
     */
    public static LDAData readData(BufferedReader reader) {
        try {
            LDAData data = new LDAData();
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                data.setDoc(line, i);
                ++i;
            }
            return data;
        }
        catch (Exception e){
            System.out.println("Read Data Error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * read a data from a string, create new dictionary
     * @param strs String from which we get the data, documents are separated by newline character
     * @return data if success and null otherwise
     */
    public static LDAData readData(String [] strs) {
        LDAData data = new LDAData(strs.length);
        for (int i = 0 ; i < strs.length; ++i) {
            data.setDoc(strs[i], i);
        }
        return data;
    }
}
