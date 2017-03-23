"""
Expected format:
...
annotations<TAB>an not at ion s, ...
...
"""
import sys
import morph_io as mio


def output_segments(segmentation: str):
    """
    :param segmentation: ex.: "abound ed"
    """

    # Start of word
    print(mio.START_SIGN + mio.SEPARATOR + mio.START)

    # Actual word
    for segment in segmentation.split(' '):
        if len(segment) == 1:
            print(segment + mio.SEPARATOR + mio.SINGLE)
        else:
            print(segment[0] + mio.SEPARATOR + mio.BEGIN)
            for index in range(1, len(segment) - 1):
                print(segment[index] + mio.SEPARATOR + mio.MIDDLE)
            print(segment[len(segment)-1] + mio.SEPARATOR + mio.END)

    # End of word
    print(mio.STOP_SIGN + mio.SEPARATOR + mio.STOP)


def process_line(line: str):
    """
    :param line: ex.: "bowler<TAB>bowl er, bowler"
    """

    segmentation_set = {x.strip(' \n') for x in line.split('\t')[1].split(',')}
    for segmentation in segmentation_set:
        output_segments(segmentation)


def main():
    if len(sys.argv[1:]) == 1:
        for line in mio.read_file(sys.argv[1:][0]):
            process_line(line)
    else:
        for line in mio.read_stdin():
            process_line(line)


if __name__ == "__main__":
    main()
