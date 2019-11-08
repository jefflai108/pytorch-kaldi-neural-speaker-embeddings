import os
import sys

# wav.scp
with open(os.path.join(sys.argv[2], 'wav.scp'), 'w') as f:
    for root, directory, files in os.walk(sys.argv[1]):
        for file in files:
            utt = file[:-4].split('_')[0]
            seg = file[:-4].split('_')[1]
            key = utt + '-' + utt + '-' + seg
            rxfile = os.path.join(root, file)
            #f.write('%s %s\n' % (key, rxfile))
            f.write('%s sox %s -t wav -c 1 -r 16000 -b 16 -e signed-integer - |\n' % (key, rxfile))

# utt2spk
with open(os.path.join(sys.argv[2], 'utt2spk'), 'w') as f:
    for root, directory, files in os.walk(sys.argv[1]):
        for file in files:
            utt = file[:-4].split('_')[0]
            seg = file[:-4].split('_')[1]
            key = utt + '-' + utt + '-' + seg
            f.write('%s %s\n' % (key, utt))
