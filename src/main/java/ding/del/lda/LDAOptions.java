package ding.del.lda;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class LDAOptions {

    String modelName = "";
    String dir = "";
    String dataFile = "";
    String vocabularyFile = "";
    double alpha = -1.0;
    double beta = -1.0;
    int topicNum = 100;
    int iterationNum = 1000;
    int saveStep = 100; //number of steps to save the model since the last save
    int topWords = 10; //the number of most likely words to be printed for each topic

    /**
     *
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
            }

            if (line.hasOption("dfile")) {
                this.dataFile = line.getOptionValue("dfile");
            }

            if (line.hasOption("vfile")) {
                this.vocabularyFile = line.getOptionValue("vfile");
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

            if (line.hasOption("savestep")) {
                this.saveStep = Integer.parseInt(line.getOptionValue("savestep"));
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

        Option dfile = OptionBuilder.withArgName("file").hasArg().withDescription("data file").create("dfile");
        options.addOption(dfile);

        Option vfile = OptionBuilder.withArgName("file").hasArg().withDescription("vocabulary file").create("vfile");
        options.addOption(vfile);

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

        Option savestep = OptionBuilder.withArgName("number").hasArg().withDescription(
                "number of steps to save the model since the last save").create("savestep");
        options.addOption(savestep);

        Option twords = OptionBuilder.withArgName("number").hasArg().withDescription(
                "number of most likely words to be displayed for each topic").create("twords");
        options.addOption(twords);

        return options;
    }
}