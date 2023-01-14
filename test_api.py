with open("./utils/config.txt", "r") as file:
    line = file.readlines()
    api = line[0].split(sep=" = ")[-1]

print(api)