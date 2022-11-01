def hp_write_in_file(path_to_file, data):
    with open(path_to_file, "w+") as f:
        for item in data:
            f.write("%s\n" % item)
