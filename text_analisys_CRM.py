from methods.text_analysis_methods import *
import getopt
import sys 

def main():
    """
    :parameter: --train/--classify : se fare classificazione o predire testi
    :parameter: --in_path : path dove sono situati i testi per training/classificazione
    :parameter: --out_path : cartella di
                salvataggio dei risultati
    """
    train = False
    predict = False
    int_label = False
    parallel = False
    out_path = in_path = 'None'
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'ti:co:p', ['train', 'in_path=', 'classify', 'out_path=', 'int_label', 'parallel'])
        for opt, arg in options:
            if opt in ('-t', '--train'):
                train = True
            elif opt in ('-c', '--classify'):
                predict = True
            elif opt in ('-i', '--in_path'):
                in_path = arg
            elif opt in ('-o', '--out_path'):
                out_path = arg
            elif opt in  ('-l', ',--int_label'):
                int_label = True
            elif opt in ('-p', ',--parallel'):
                parallel = True

        if train:
            train_classifiers(in_path, out_path, parallel)
        elif predict:
            predict_label(in_path)

    except getopt.GetoptError as err:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    # ...


if __name__ == "__main__":
    main()