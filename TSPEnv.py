import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import random
from Test_data.test import load_instances_with_baselines


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    data: torch.Tensor


class TSPEnv:
    def __init__(self, **env_params):

        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params['data_path']
        self.sub_path = env_params['sub_path']
        self.batch_size = None
        self.problems = None
        self.raw_data_nodes = []
        self.raw_data_tours = []
        self.raw_data_nodes_100 = []
        self.raw_data_tours_100 = []
        self.selected_count = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.episode = None
        self.solution = None

    def load_problems(self, episode, batch_size, mix=False, train=False):
        self.episode = episode
        self.batch_size = batch_size

        if mix:
            index = random.sample(range(1000000), batch_size)
            problems_small, solution_small = self.raw_data_nodes_100[index], self.raw_data_tours_100[index]
            problems_large, solution_large = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[
                                                                                                episode:episode + batch_size]
            if self.sub_path:
                problems_large, solution_large = self.sampling_subpaths(problems_large, solution_large, mode='train',
                                                                        low_index=101)

            if_inverse = True
            if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]
            if if_inverse_index < 50:
                if_inverse = False
            if if_inverse:
                solution_large = torch.flip(solution_large, dims=[1])

            self.problem_size = problems_large.shape[1]
            node_gap = self.problem_size - 100
            index_bsz = torch.arange(batch_size, dtype=torch.long, device=problems_large.device)
            cor = problems_small[index_bsz, solution_small[:, 0]].unsqueeze(1).repeat(1, node_gap, 1)
            problems_small = torch.cat((cor, problems_small), dim=1)
            sol = torch.arange(node_gap, dtype=torch.long, device=problems_large.device)[None, :].repeat(batch_size, 1)
            solution_small = torch.cat((sol, solution_small + node_gap), dim=1)
            self.problems = torch.cat((problems_small, problems_large), dim=0)
            self.solution = torch.cat((solution_small, solution_large), dim=0)
            self.batch_size = batch_size * 2
        else:
            self.problems, self.solution = self.raw_data_nodes[episode:episode + batch_size], self.raw_data_tours[
                                                                                              episode:episode + batch_size]
            # shape: [B,V,2]  ;  shape: [B,V]

            if self.sub_path:
                self.problems, self.solution = self.sampling_subpaths(self.problems, self.solution, mode='train')
            if_inverse = True
            if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]
            if if_inverse_index < 50:
                if_inverse = False

            if if_inverse:
                self.solution = torch.flip(self.solution, dims=[1])
            self.problem_size = self.problems.shape[1]

        if train:
            if_rotation = torch.randint(low=0, high=8, size=[1])[0]
            x = self.problems[:, :, [0]]
            y = self.problems[:, :, [1]]

            if if_rotation == 0:
                self.problems = torch.cat((x, y), dim=2)
            elif if_rotation == 1:
                self.problems = torch.cat((1 - x, y), dim=2)
            elif if_rotation == 2:
                self.problems = torch.cat((x, 1 - y), dim=2)
            elif if_rotation == 3:
                self.problems = torch.cat((1 - x, 1 - y), dim=2)
            elif if_rotation == 4:
                self.problems = torch.cat((y, x), dim=2)
            elif if_rotation == 5:
                self.problems = torch.cat((1 - y, x), dim=2)
            elif if_rotation == 6:
                self.problems = torch.cat((y, 1 - x), dim=2)
            elif if_rotation == 7:
                self.problems = torch.cat((1 - y, 1 - x), dim=2)

    def sampling_subpaths(self, problems, solution, length_fix=False, mode='test', repair=False, low_index=4):
        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        embedding_size = problems.shape[2]
        first_node_index = torch.randint(low=0, high=problems_size, size=[1])[0]  # in [0,N)
        if mode == 'test':
            length_of_subpath = torch.randint(low=low_index, high=problems_size + 1, size=[1])[0]  # in [4,N]
        else:
            if length_fix:
                length_of_subpath = problems_size
            else:
                length_of_subpath = torch.randint(low=low_index, high=problems_size + 1, size=[1])[0]  # in [4,N]
        double_solution = torch.cat([solution, solution], dim=-1)
        new_sulution = double_solution[:, first_node_index: first_node_index + length_of_subpath]
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        index_2, _ = torch.cat((new_sulution_ascending, new_sulution_ascending), dim=1).type(torch.long).sort(dim=-1,
                                                                                                              descending=False)  # shape: [B, 2current_step]
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[
            1])  # shape: [B, 2current_step]
        temp = torch.arange((embedding_size), dtype=torch.long)[None, :].expand(batch_size,
                                                                                embedding_size)  # shape: [B, current_step]
        index_3 = temp.repeat([1, length_of_subpath])

        new_data = problems[index_1, index_2, index_3].view(batch_size, length_of_subpath, 2)

        if repair == True:
            return new_data, new_sulution_rank, first_node_index, length_of_subpath, double_solution
        else:
            return new_data, new_sulution_rank

    def shuffle_data(self):
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]

    def load_problems_4_each_epoch(self, batch_size, problem_size):
        self.batch_size = batch_size
        self.problem_size = problem_size
        self.raw_data_nodes = torch.rand(size=(batch_size, problem_size, 2), device=torch.device('cuda'),
                                         requires_grad=False)
        self.raw_data_tours = None

    def load_problems_val(self, episode, batch_size, if_rotation=0):
        self.problems = self.raw_data_nodes[episode:episode + batch_size]
        if if_rotation != 0:
            x = self.problems[:, :, [0]]
            y = self.problems[:, :, [1]]
            if if_rotation == 1:
                self.problems = torch.cat((1 - x, y), dim=2)
            elif if_rotation == 2:
                self.problems = torch.cat((x, 1 - y), dim=2)
            elif if_rotation == 3:
                self.problems = torch.cat((1 - x, 1 - y), dim=2)
            elif if_rotation == 4:
                self.problems = torch.cat((y, x), dim=2)
            elif if_rotation == 5:
                self.problems = torch.cat((1 - y, x), dim=2)
            elif if_rotation == 6:
                self.problems = torch.cat((y, 1 - x), dim=2)
            elif if_rotation == 7:
                self.problems = torch.cat((1 - y, 1 - x), dim=2)
        self.solution = None

    def load_raw_data(self, episode, begin_index=0, SL_Test=True, MSV=False, size=None, disribution=None):
        print('load raw dataset begin!')
        if SL_Test:
            if MSV:
                root = f"./Test_data/"
                instances, b_tours, _ = load_instances_with_baselines(root, size, disribution)
                self.raw_data_nodes = instances
                self.raw_data_tours = b_tours
            else:
                self.raw_data_nodes = []
                self.raw_data_tours = []
                for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index:episode + begin_index],
                                 ascii=True):
                    line = line.split(" ")
                    num_nodes = int(line.index('output') // 2)
                    nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

                    self.raw_data_nodes.append(nodes)
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

                    self.raw_data_tours.append(tour_nodes)

                self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
                self.raw_data_tours = torch.tensor(self.raw_data_tours, requires_grad=False)
                print(f'load raw dataset done!', )
        else:
            self.raw_data_nodes_100 = []
            self.raw_data_tours_100 = []
            for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index:episode + begin_index], ascii=True):
                line = line.split(" ")
                num_nodes = int(line.index('output') // 2)
                nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
                self.raw_data_nodes_100.append(nodes)
                tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
                self.raw_data_tours_100.append(tour_nodes)
            self.raw_data_nodes_100 = torch.tensor(self.raw_data_nodes_100, requires_grad=False)
            self.raw_data_tours_100 = torch.tensor(self.raw_data_tours_100, requires_grad=False)
            print(f'load raw dataset done!', )


    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
            self.selected_node_list = torch.zeros((batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((batch_size, 0), dtype=torch.long)
        else:
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
            self.selected_student_list = torch.zeros((self.batch_size, 0), dtype=torch.long)
        self.selected_count = 0
        self.step_state = Step_State(data=self.problems)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        reward_student = None
        done = False
        return self.step_state, reward, reward_student, done

    def step(self, selected, selected_student):

        self.selected_count += 1

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]),
                                            dim=1)  # shape: [B, current_step]

        self.selected_student_list = torch.cat((self.selected_student_list, selected_student[:, None]), dim=1)

        done = (self.selected_count == self.problems.shape[1])
        if done:
            reward_student = self._get_travel_distance_2(self.problems, self.selected_student_list)
            reward = self._get_travel_distance_2(self.problems, self.selected_node_list)
        else:
            reward, reward_student = None, None

        return self.step_state, reward, reward_student, done

    def step_beam(self, selected, beam=16):

        self.selected_count += 1

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]),
                                            dim=1)  # shape: [B, current_step]

        done = (self.selected_count == self.problems.shape[1])
        if done:
            problems = torch.repeat_interleave(self.problems, beam, 0)
            reward = self._get_travel_distance_2(problems, self.selected_node_list)
        else:
            reward = None
        return self.step_state, reward, done

    def make_dir(self, path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return

    def drawPic(self, arr_, tour_, name='xx', optimal_tour_=None, index=None):
        arr = arr_[index.item()].clone().cpu().numpy()
        tour = tour_[index.item()].clone().cpu().numpy()
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        arr = (arr - arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20))

        plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

        plt.axis('off')

        start = [arr[tour[0], 0], arr[tour[-1], 0]]
        end = [arr[tour[0], 1], arr[tour[-1], 1]]
        plt.plot(start, end, color='red', linewidth=2, )

        for i in range(len(tour) - 1):
            tour = np.array(tour, dtype=int)
            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start, end, color='red', linewidth=2)

        b = os.path.abspath(".")
        path = b + '/figure'
        self.make_dir(path)
        plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)

    def _get_travel_distance(self):
        if self.solution is not None:
            gathering_index = self.solution.unsqueeze(2).expand(self.batch_size, self.problems.shape[1], 2)
            seq_expanded = self.problems
            ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2)
            segment_lengths = segment_lengths.sum(2).sqrt()
            travel_distances = segment_lengths.sum(1)

            # trained model's distance
            gathering_index_student = self.selected_student_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
            ordered_seq_student = self.problems.gather(dim=1, index=gathering_index_student)
            rolled_seq_student = ordered_seq_student.roll(dims=1, shifts=-1)
            segment_lengths_student = ((ordered_seq_student - rolled_seq_student) ** 2)
            segment_lengths_student = segment_lengths_student.sum(2).sqrt()
            # shape: (batch,problem)
            travel_distances_student = segment_lengths_student.sum(1)
            # shape: (batch)
            return travel_distances, travel_distances_student
        else:
            return 0, 0

    def get_travel_distance_multi(self, selected_node_list):
        batch_size, problem_size = selected_node_list.size(0), selected_node_list.size(2)
        gathering_index = selected_node_list.unsqueeze(3).expand(batch_size, -1, problem_size, 2)
        # shape: (batch, num_multi, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(batch_size, selected_node_list.size(1), problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, num_multi, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, num_multi problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, num_multi)
        return travel_distances


    def _get_travel_distance_2(self, problems, solution):

        gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

        seq_expanded = problems

        ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

        segment_lengths = ((ordered_seq - rolled_seq) ** 2)

        segment_lengths = segment_lengths.sum(2).sqrt()

        travel_distances = segment_lengths.sum(1)

        return travel_distances
