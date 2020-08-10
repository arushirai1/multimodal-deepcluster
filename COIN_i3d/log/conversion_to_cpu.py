import pickle

file_paths=['prec5', 'prec1']# 'loss_log']
files=[open(f, 'rb') for f in file_paths]

loaded_files=[pickle.load(f) for f in files]
cpu_formatted={file_name:[tensor.item() for tensor in loaded_files[i]] for i, file_name in enumerate(file_paths)}
cpu_file = open('cpu_formatted', 'wb')
pickle.dump(cpu_formatted, cpu_file)
for f in files:
    f.close()

cpu_file.close()
