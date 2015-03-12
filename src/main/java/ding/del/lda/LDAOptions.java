package ding.del.lda;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.File;

public class LDAOptions {

  String modelName = "unnamed-model";
  String dir = "./";
  String cfile = "documents.txt"; // corpus file
  String vfile = "vocabulary.txt"; // vocabulary file
  String sfile = "stopwords.txt"; // stopwords file

  int topicNum = 100;
  double alpha = 50.0 / topicNum;
  double beta = 0.1;


  int iterationNum = 500; // number of Gibbs sampling iteration
  int burnIn = 100; // number of iterations for burn-in period
  int saveInterval = 100; // saving period
  int topWords = 10; // the number of most likely words to be printed for each topic

  /**
   * @param args command line arguments
   */
  public LDAOptions(String[] args) {
    Options options = buildOptions();
    CommandLineParser parser = new GnuParser();
    CommandLine line;
    try {
      line = parser.parse(options, args);

      if (line.hasOption("modelname")) {
        this.modelName = line.getOptionValue("modelname");
      }

      if (line.hasOption("dir")) {
        this.dir = line.getOptionValue("dir");
        if (this.dir.endsWith(File.separator))
          this.dir = this.dir.substring(0, this.dir.length() - 1);
      }

      if (line.hasOption("cfile")) {
        this.cfile = line.getOptionValue("cfile");
      }

      if (line.hasOption("vfile")) {
        this.vfile = line.getOptionValue("vfile");
      }

      if (line.hasOption("sfile")) {
        this.sfile = line.getOptionValue("sfile");
      }

      if (line.hasOption("alpha")) {
        this.alpha = Double.parseDouble(line.getOptionValue("alpha"));
      }

      if (line.hasOption("beta")) {
        this.beta = Double.parseDouble(line.getOptionValue("beta"));
      }

      if (line.hasOption("ntopics")) {
        this.topicNum = Integer.parseInt(line.getOptionValue("ntopics"));
      }

      if (line.hasOption("niters")) {
        this.iterationNum = Integer.parseInt(line.getOptionValue("niters"));
      }

      if (line.hasOption("burn")) {
        this.burnIn = Integer.parseInt(line.getOptionValue("burn"));
      }

      if (line.hasOption("saveInterval")) {
        this.saveInterval = Integer.parseInt(line.getOptionValue("saveInterval"));
      }

      if (line.hasOption("twords")) {
        this.topWords = Integer.parseInt(line.getOptionValue("twords"));
      }
    } catch (ParseException e) {
      System.err.println("Parsing failed: " + e.getMessage());
    }
  }

  @SuppressWarnings("static-access")
  private static Options buildOptions() {
    Options options = new Options();

    Option modelname = OptionBuilder.withArgName("name").hasArg().withDescription("model name").create("modelname");
    options.addOption(modelname);

    Option dir = OptionBuilder.withArgName("path").hasArg().withDescription("directory").create("dir");
    options.addOption(dir);

    Option cfile = OptionBuilder.withArgName("file").hasArg().withDescription("corpus file").create("cfile");
    options.addOption(cfile);

    Option vfile = OptionBuilder.withArgName("file").hasArg().withDescription("vocabulary file").create("vfile");
    options.addOption(vfile);

    Option sfile = OptionBuilder.withArgName("file").hasArg().withDescription("stopwords file").create("sfile");
    options.addOption(sfile);

    Option alpha = OptionBuilder.withArgName("value").hasArg().withDescription("alpha").create("alpha");
    options.addOption(alpha);

    Option beta = OptionBuilder.withArgName("value").hasArg().withDescription("beta").create("beta");
    options.addOption(beta);

    Option ntopics = OptionBuilder.withArgName("number").hasArg().withDescription(
            "number of topics").create("ntopics");
    options.addOption(ntopics);

    Option niters = OptionBuilder.withArgName("number").hasArg().withDescription(
            "number of iterations").create("niters");
    options.addOption(niters);

    Option burn = OptionBuilder.withArgName("number").hasArg().withDescription(
            "number of iterations for burn-in period").create("burn");
    options.addOption(burn);

    Option saveInterval = OptionBuilder.withArgName("number").hasArg().withDescription(
            "number of steps to save the model since the last save").create("saveInterval");
    options.addOption(saveInterval);

    Option twords = OptionBuilder.withArgName("number").hasArg().withDescription(
            "number of most likely words to be displayed for each topic").create("twords");
    options.addOption(twords);

    return options;
  }
}