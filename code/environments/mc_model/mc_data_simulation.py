import argparse
import datetime
import json
import pickle
import os

from mc_lob_simulation_class import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file', type=str, required=True)

    return parser.parse_args()


def main(args):
    # load info file
    with open(os.getcwd() + "\\" + args.info_file, 'r') as f:
        info = json.load(f)

    # set seed
    np.random.seed(info["seed"])

    # model info
    info["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    # set up a start LOB
    x0 = np.zeros((2, info["num_levels"] + 1), dtype=int)
    x0[0, 1:] = 1
    x0[1, :] = -1

    mc_model = MarkovChainLobModel(num_levels=info["num_levels"], ob_start=x0)

    # add rates as lists to info file as well
    info["rates"] = {}
    for k, v in mc_model.rates.items():
        if isinstance(v, np.ndarray):
            info["rates"][k] = list(v)
        else:
            info["rates"][k] = v
    info["event types"] = mc_model.event_types

    # save info file
    with open(info["data_path"] + "//" + info["filename"] + '_info.txt', 'w') as file:
        file.write(json.dumps(info))

    # simulate away from initial
    _ = mc_model.simulate(num_events=info["num_initial_events"])

    # simulate events and save
    data_dict = mc_model.simulate(num_events=info["num_events"])
    data_dict["event_types"] = mc_model.event_types

    pickle.dump(data_dict, open("{}//{}.pkl".format(info["data_path"], info["filename"]), "wb"))


if __name__ == '__main__':
    main(parse_args())
