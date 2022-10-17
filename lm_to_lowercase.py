with open('3-gram.pruned.1e-7.arpa', 'r') as f:
    with open('lm.arpa', 'w') as f_out:
        for line in f:
            f_out.write(line.lower())
