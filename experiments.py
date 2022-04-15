from config_experiments import parameters
import itertools
import os

"""
def experiment():
    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    with open(os.path.join(os.path.dirname(__file__), 'exps.bash'), 'w') as f:
        f.write("#!*bash*" + '\n')
        for d in permutations_dicts:
            infos = "--problem " + d["problem"] + " --model " + d["model"] + " --ntasks " + str(d["ntasks"]) + \
                    " --lam_reg " + str(d["lam_reg"]) + " --num_outer_steps " + str(d["epochs"]) + " --trainer " + d["trainer"]
            train = "python " + os.path.join(os.path.dirname(__file__)) + "train_synthetic_sparsity.py " + infos
            test = "python " + os.path.join(os.path.dirname(__file__)) + "test.py " + infos + " < 1"
            f.write(train + '\n')
            f.write(test + '\n')
"""


def run_experiments():
    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, d in enumerate(permutations_dicts):
        infos = "--problem " + d["problem"] + " --model " + d["model"] + " --ntasks " + str(d["ntasks"]) + \
                " --lam_reg " + str(d["lam_reg"]) + " --num_outer_steps " + str(d["epochs"]) + " --trainer " + d[
                    "trainer"]
        train = "python " + os.path.join(os.path.dirname(__file__)) + "train_synthetic_sparsity.py " + infos
        test = "python " + os.path.join(os.path.dirname(__file__)) + "test.py " + infos
        print("run " + str(i) + " : parameters " + str(d))
        os.system(train)
        os.system(test)


def generate_datasets():
    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for p in parameters["problem"]:
        for n in parameters["ntasks"]:
            line = "python generate_synthetic_data.py --problem " + p + " --ntasks " + str(n)
            print("Generating problem " + p + " with "  + str(n)  + " tasks.")
            os.system(line)


if __name__ == "__main__":
    generate_datasets()
    # run_experiments()
