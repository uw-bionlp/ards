import os
import sys
import json
from pathlib import Path
from pprint import pprint
from argparse import ArgumentParser
from time import strftime

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file_or_dir', help='Absolute or relative path to the file or directory to parse.')
    parser.add_argument('--output_path', help='Directory or .jsonl file to write output to. Defaults to /output/<current_time>/')
    parser.add_argument('--output_single_file', help='Write output to a single .jsonl file with one line per document output', default=False, dest='output_single_file', action='store_true')
    parser.add_argument('--batch_size', help='Number of documents to process at a time.', default=8, type=int)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    args.file_or_dir = os.path.abspath(args.file_or_dir)
    if not args.output_path:
        args.output_path = os.path.join(os.path.dirname(__file__), 'output', f'{Path(args.file_or_dir).stem}_{strftime("%Y%m%d-%H%M%S")}')


    print('Starting up ARDS predictor! Params:\n')
    pprint(vars(args))

    if not os.path.exists(args.file_or_dir):
        print(f"The file or directory '{args.file_or_dir}' could not be found!\n")
        sys.exit()

    print(f"\Predictions will be output to '{args.output_path}'")

    return args


def write_output(output, filename, args):
    """ Write output to directory or .jsonl file in pretty-printed JSON """

    # Single .jsonl file
    if args.output_single_file:
        output['id'] = filename + '.txt'
        filename = os.path.join(args.output_path, 'output.jsonl')
        if not os.path.exists(filename):
            with open(filename, 'w+') as f: f.write('')
        with open(filename, 'a+') as f:
            f.write(json.dumps(output) + '\n')

    # One .json file per document
    else:
        with open(os.path.join(args.output_path, filename + '.json'), 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


def batch_files(files, args):
    if args.batch_size == 0:
        return [ files ]

    batches, curr_batch, i, files_len = [], [], 0, len(files)
    while i < files_len:
        curr_batch.append(files[i])
        if len(curr_batch) == args.batch_size:
            batches.append(curr_batch)
            curr_batch = []
        i += 1
    if len(curr_batch):
        batches.append(curr_batch)
    return batches


def main():
    """ Run the client """

    # Parse args, bail if invalid
    args = parse_args()
    
    print("Loading model...")
    from process import DocumentProcessor
    processor = DocumentProcessor()

    # Make output directory
    if not os.path.exists(args.output_path):
        Path(args.output_path).mkdir(parents=True)

    # Load documents
    if os.path.isfile(args.file_or_dir):
        files = [ args.file_or_dir ]
        print(f"Found 1 text file '{args.file_or_dir}'\n")
    else:
        files = [ os.path.join(args.file_or_dir, f) for f in os.listdir(args.file_or_dir) if Path(f).suffix == '.txt' ]
        print(f"Found {len(files)} text file(s) in '{args.file_or_dir}'\n")

    existing_files = [ f for f in os.listdir(args.output_path) if Path(f).suffix == '.json' ]
    if any(existing_files) and not args.output_single_file:
        overlap = [ f for f in files if Path(f).stem+'.json' in existing_files ]
        if any(overlap):
            print(f"Found {len(existing_files)} existing json files, these will be skipped")       
            files = [ f for f in files if f not in overlap ]

    if not any(files):
        print(f"There are no files to process!")  
        sys.exit()

    # Loop through batches
    batches = batch_files(files, args)
    X = []
    for batch in batches:
        for f in batch:
            with open(f, 'r') as fin:
                X.append(fin.read())
        
        # Predict labels
        Y = processor.predict(X)

        # Output to file(s)
        for file, y in zip(batch, Y):
            write_output(y, Path(file).stem, args)

    print(f"All done! Results written to '{args.output_path}'") 


if __name__ == '__main__':
    main()