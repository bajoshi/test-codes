import numpy as np

import sys

def main():

    counter = int(sys.argv[1])
    #print("Received counter from bash:", counter)

    a = np.arange(100, dtype=np.float64)
    a = a.reshape((10,10))

    file_save_name = 'test_arr_for_bash_script_' + str(counter) + '.npy'
    np.save(file_save_name, a)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)