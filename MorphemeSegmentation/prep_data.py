'''
Expected format:
...
annotations<TAB>an not at ion s, ...
...
'''
import sys


START_SIGN='<s>'
START='START'
BEGIN='B'
MIDDLE='M'
END='E'
SINGLE='S'
STOP='STOP'
STOP_SIGN='</s>'
SEPARATOR='\t'


def output_segments(segmentation: str):
    # Start of word
    print(START_SIGN + SEPARATOR + START)
    # Actual word
    for segment in segmentation.split(' '):
        if len(segment) == 1:
            print(segment + SEPARATOR + SINGLE)
        else:
            print(segment[0] + SEPARATOR + BEGIN)
            for index in range(1, len(segment) - 1):
                print(segment[index] + SEPARATOR + MIDDLE)
            print(segment[len(segment)-1] + SEPARATOR + END)
    # End of word
    print(STOP_SIGN + SEPARATOR + STOP)


def process_line(line: str):
    segmentation_set = {x.strip(' \n') for x in line.split('\t')[1].split(',')}
    for segmentation in segmentation_set:
        output_segments(segmentation)


def read_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            yield line


def read_stdin():
    for line in sys.stdin:
        yield line


def main():
    if len(sys.argv[1:]) == 1:
        for line in read_file(sys.argv[1:][0]):
            process_line(line)
    else:
        for line in read_stdin():
            process_line(line)


if __name__ == "__main__":
    main()
