package ding.del.lda;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class Vocabulary {
    Map<String, Integer> wordToIndex;
    Map<Integer, String> indexToWord;

    public Vocabulary(){
        wordToIndex = new HashMap<String, Integer>();
        indexToWord = new HashMap<Integer, String>();
    }

    public String getWord(int index){
        return indexToWord.get(index);
    }

    public Integer getIndex(String word){
        return wordToIndex.get(word);
    }

    public boolean contains(String word){
        return wordToIndex.containsKey(word);
    }

    public boolean contains(int index){
        return indexToWord.containsKey(index);
    }

    public int addWord(String word){
        if (!contains(word)){
            int index = wordToIndex.size();

            wordToIndex.put(word, index);
            indexToWord.put(index, word);

            return index;
        }
        else return getIndex(word);
    }

    /**
     * read vocabulary from file
     * each line contains a word and its index
     */
    public boolean readVocabulary(String vocabularyFile) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(vocabularyFile), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split("\\s");
                String word = tokens[0];
                int index = Integer.parseInt(tokens[1]);
                indexToWord.put(index, word);
                wordToIndex.put(word, index);
            }
            reader.close();
            return true;
        }
        catch (Exception e) {
            System.out.println("Error while reading vocabulary: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    public boolean writeVocabulary(String vocabularyFile){
        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(vocabularyFile), "UTF-8"));

            for (String word : wordToIndex.keySet()) {
                Integer index = wordToIndex.get(word);
                writer.write(word + " " + index + "\n");
            }
            writer.close();
            return true;
        }
        catch (Exception e){
            System.out.println("Error while writing vocabulary: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}