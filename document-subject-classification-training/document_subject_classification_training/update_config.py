import json
from argparse import ArgumentParser

def main(args):
    data_root_path = args.data_path

    with open("config.json", "r") as jsonFile:
        data = json.load(jsonFile)

    data["custom_data_path"] = data_root_path

    with open("config.json", "w") as jsonFile:
        json.dump(data, jsonFile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', help='path to the data folder', type=str)
    main(parser.parse_args())