import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
            description = 'Creating the label list for OpenSphere Training datasets')
    parser.add_argument('--dataset_dir', 
            help = 'the dir to the dataset for creating the label list')
    parser.add_argument('--list_name', 
            help = 'the file name to save the label list')
    args = parser.parse_args()
    return args


def ListFilesToTxt(dir,file,wildcard,recursion):
    global last_name
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            last_name = name
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    file.write( fullname + " " + last_name + "\n")
                    break

if __name__ == '__main__':
    args = parse_args()
    if args.dataset_dir:
        dir = args.dataset_dir
        outfile = args.list_name + ".txt"
        wildcard = ".jpg .jpeg .png"
        file = open(outfile, "w")
        if not file:
            print ("cannot open the file %s for writing" % outfile)
        ListFilesToTxt(dir, file, wildcard, 1)
        file.close()
    else:
        print('No dataset directory is provided')

