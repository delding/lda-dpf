if __name__ == "__main__":
    with open("ap.txt") as input_file:
        with open("ap_documents.txt", 'w') as output_file:
            for line in input_file:
                if len(line.strip()) > 0 and line.strip()[0] != "<":
                    output_file.write(line.strip() + '\n')
