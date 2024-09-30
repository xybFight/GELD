import torch
from pathlib import Path


# This code is based on INViT

def read_solutions_from_file(file_path):
    tour_storage = []
    tour_len_storage = []
    ellapsed_time_storage = []
    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            tour_text, tour_len_text, ellapsed_time_text = line_text.strip().split(" ")

            tour = [int(val) for val in tour_text.split(",")]
            tour_storage.append(tour)

            tour_len = float(tour_len_text)
            tour_len_storage.append(tour_len)

            ellapsed_time = float(ellapsed_time_text)
            ellapsed_time_storage.append(ellapsed_time)

            line_text = read_file.readline()

    tours = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tour_storage], batch_first=True, padding_value=0)
    tour_lens = torch.tensor(tour_len_storage)
    time_consumptions = torch.tensor(ellapsed_time_storage)
    return tours, tour_lens, time_consumptions


def read_tsp_instances_from_file(file_path):
    """
    read instances from the given file (should follow the rules in write_tsp_instances_to_file())
    :param file_path: the input data path
    :return: a (num, size, 2) tensor in cpu, multiple tsp instances
    """
    tsp_instances = []
    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            splitted_text = line_text.strip().split(" ")
            tsp_instance = []
            for node_text in splitted_text:
                tsp_instance.append([float(val) for val in node_text.split(",")])
            tsp_instances.append(tsp_instance)
            line_text = read_file.readline()
    return torch.Tensor(tsp_instances)


def read_tour_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    tour = []
    reading_tour = False
    for line in lines:
        if line.startswith('TOUR_SECTION'):
            reading_tour = True
            continue
        elif line.startswith('EOF'):
            break
        if reading_tour:
            node = int(line.strip())
            tour.append(node)
    tour_tensor = torch.tensor(tour, dtype=torch.int64)
    tour_tensor = tour_tensor - 1
    tour_tensor = tour_tensor[:-1]
    return tour_tensor


def load_tsp_instances_with_baselines(root, size, distribution):
    baseline = "None"
    if size == 100:
        baseline = "LKH3_runs10"
    elif size == 500:
        baseline = "LKH3_runs10"
    elif size == 1000:
        baseline = "LKH3_runs10"
    elif size == 5000:
        baseline = "LKH3_runs1"
    elif size == 10000:
        baseline = "LKH3_runs1"

    instance_root = Path(root)
    instance_dir = f"data_farm/tsp{size}/"
    instance_name = f"tsp{size}_{distribution}.txt"
    instance_file = instance_root.joinpath(instance_dir).joinpath(instance_name)

    tsp_instances = read_tsp_instances_from_file(instance_file)

    solution_root = Path(root)
    solution_dir = f"solution_farm/tsp{size}_{distribution}/"
    solution_name = f"{baseline}.txt"
    solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)
    baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)
    return tsp_instances, baseline_tours, baseline_lens


def load_instances_with_baselines(root, size, distribution):
    return load_tsp_instances_with_baselines(root, size, distribution)


# the lens is derived directly from TSPLIB
tsplib_collections = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'pr76': 108159,
    'eil76': 538,
    'rat99': 1211,
    'kroA100': 21282,
    'kroE100': 22068,
    'kroB100': 22141,
    'rd100': 7910,
    'kroD100': 21294,
    'kroC100': 20749,
    'eil101': 629,
    'lin105': 14379,
    'pr107': 44303,
    'pr124': 59030,
    'bier127': 118282,
    'ch130': 6110,
    'pr136': 96772,
    'pr144': 58537,
    'kroA150': 26524,
    'kroB150': 26130,
    'ch150': 6528,
    'pr152': 73682,
    'u159': 42080,
    'rat195': 2323,
    'd198': 15780,
    'kroA200': 29368,
    'kroB200': 29437,
    'tsp225': 3916,
    'ts225': 126643,
    'pr226': 80369,
    'gil262': 2378,
    'pr264': 49135,
    'a280': 2579,
    'pr299': 48191,
    'lin318': 42029,
    'rd400': 15281,
    'fl417': 11861,
    'pr439': 107217,
    'pcb442': 50778,
    'd493': 35002,
    'u574': 36905,
    'rat575': 6773,
    'p654': 34643,
    'd657': 48912,
    'u724': 41910,
    'rat783': 8806,
    'pr1002': 259045,
    'u1060': 224094,
    'vm1084': 239297,
    'pcb1173': 56892,
    'd1291': 50801,
    'rl1304': 252948,
    'rl1323': 270199,
    'nrw1379': 56638,
    'fl1400': 20127,
    'u1432': 152970,
    'fl1577': 22249,
    'd1655': 62128,
    'vm1748': 336556,
    'u1817': 57201,
    'rl1889': 316536,
    'd2103': 80450,
    'u2152': 64253,
    'u2319': 234256,
    'pr2392': 378032,
    'pcb3038': 137694,
    'fl3795': 28772,
    'fnl4461': 182566,
    'rl5915': 565530,
    'rl5934': 556045,
    'rl11849': 923288,
    'usa13509': 19982859,
    'brd14051': 469385,
    'd15112': 1573084,
    'd18512': 645238
}
# the lens is derived directly from World TSP datasets
national_collections = {
    'WI29': 27603,
    'DJ38': 6656,
    'QA194': 9352,
    'UY734': 79114,
    'ZI929': 95345,
    'LU980': 11340,
    'RW1621': 26051,
    'MU1979': 86891,
    'NU3496': 96132,
    'CA4663': 1290319,
    'TZ6117': 394609,
    'EG7146': 172386,
    'YM7663': 238314,
    'PM8079': 114831,
    'EI8246': 206128,
    'AR9152': 837377,
    'JA9847': 491924,
    'GR9882': 300899,
    'KZ9976': 1061387,
    'FI10639': 520383,
    'MO14185': 427246,
    'HO14473': 176940,
    'IT16862': 557315,
    'VM22775': 569115,
    'SW24978': 855597,
    'BM33708': 959011,
    'CH71009': 4565452,
    # extremely large-scale instances
    # 'SRA104815' : 251342,
    # 'ARA238025' : 578761,
    # 'LRA498378' : 2168039,
    # 'LRB744710' : 1611232
}


def read_tsplib_file(file_path):
    """
    The read_tsplib_file function reads a TSPLIB file and returns the nodes and name of the problem.
    
    :param file_path: Specify the path to the file that is being read
    :return: A list of nodes and a name
    """
    properties = {}
    reading_properties_flag = True
    nodes = []

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            else:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    pass
                else:
                    line_contents = [x.strip() for x in line.split(" ") if x.strip()]
                    _, x, y = line_contents
                    nodes.append([float(x), float(y)])
            line = read_file.readline()

    return nodes, properties["NAME"]


def load_tsplib_file(root, tsplib_name, TSPLIB_or_Nation=False):
    if TSPLIB_or_Nation:
        tsplib_dir = "tsplib"
    else:
        tsplib_dir = "National_TSP"
    file_name = f"{tsplib_name}.tsp"
    file_path = root.joinpath(tsplib_dir).joinpath(file_name)
    instance, name = read_tsplib_file(file_path)

    instance = torch.tensor(instance)
    return instance, name

