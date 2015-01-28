package ding.del.lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;

public class Stopwords {

    HashSet<String> stopwords = new HashSet<String>();

    /**
     * read stop words from file
     */
    public void loadStopwords(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String word;
            while((word = reader.readLine()) != null) {
                add(word);
            }
            reader.close();
        }
        catch (Exception e) {
            System.err.println("Read Stop Words Error: " + e.getMessage());
        }
    }

    /**
     * removes all stop words
     */
    public void clear() {
        stopwords.clear();
    }

    /**
     * adds the given word to the stop word list (automatically converted to
     * lower case and trimmed)
     * @param word the word to add
     */
    public void add(String word) {
        if (word.trim().length() > 0)
            stopwords.add(word.trim().toLowerCase());
    }

    /**
     * removes the word from the stop word list
     * @param word the word to remove
     * @return true if the word was found in the list and then removed
     */
    public boolean remove(String word) {
        return stopwords.remove(word);
    }

    public boolean isStopword(String str) {
        return stopwords.contains(str.toLowerCase());
    }
}