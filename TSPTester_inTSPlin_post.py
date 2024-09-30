from logging import getLogger
import torch

from TSP_Model import TSPModel as Model
from TSPEnv_inTSPlib import TSPEnv as Env
from utils.utils import *
from Test_data.test import tsplib_collections, national_collections, read_tour_file


class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.dtypeFloat = torch.cuda.FloatTensor
        self.dtypeLong = torch.cuda.LongTensor

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        if not self.env_params['test_in_tsplib']:
            self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_500 = []
        problems_500_1k = []
        problems_1k_5k = []
        problems_5k_10k = []
        problems_10k_ = []

        # only nationalTSPs
        # you can change the "INV_sol.py", i.e., the solution produced by INViT, to other files, e.g., UDC.sol produced by UDC.
        ss_solutions = np.load('baseline_solutions/INV_so.npy', allow_pickle=True).item()
        print(f"Start evaluation...")
        for name in ss_solutions:
            opt_len = national_collections[name]
            score_teacher, score_student, problems_size = self._test_one_batch(episode, 1, clock=self.time_estimator_2,
                                                                               Name=name.lower(),
                                                                               opt_len=torch.tensor(float(opt_len)),
                                                                               solution=ss_solutions[name])
            current_gap = (score_student - score_teacher) / score_teacher
            if problems_size <= 100:
                problems_100.append(current_gap)
            elif problems_size <= 500:
                problems_100_500.append(current_gap)
            elif problems_size <= 1000:
                problems_500_1k.append(current_gap)
            elif problems_size <= 5000:
                problems_1k_5k.append(current_gap)
            elif problems_size <= 10000:
                problems_5k_10k.append(current_gap)
            else:
                problems_10k_.append(current_gap)

            print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            print('problems_100_500 mean gap:', np.mean(problems_100_500), len(problems_100_500))
            print('problems_500_1k mean gap:', np.mean(problems_500_1k), len(problems_500_1k))
            print('problems_1k_5k mean gap:', np.mean(problems_1k_5k), len(problems_1k_5k))
            print('problems_5k_10k mean gap:', np.mean(problems_5k_10k), len(problems_5k_10k))
            print('problems_10k_ mean gap:', np.mean(problems_10k_), len(problems_10k_))
            score_AM.update(score_teacher, 1)
            score_student_AM.update(score_student, 1)

            episode += 1
            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(1, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student, ))

            all_done = (episode == test_num_episode)

            if all_done:
                if not self.env_params['test_in_tsplib']:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

                else:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_100 + problems_100_500 + problems_500_1k + problems_1k_5k + problems_5k_10k + problems_10k_
                    average_gap = np.mean(all_result_gaps)
                    self.logger.info(" Average Gap: {:.4f}%".format(average_gap * 100))
                    gap_ = average_gap

        return score_AM.avg, score_student_AM.avg, gap_

    def _test_one_batch(self, episode, batch_size, clock=None, Name=None, opt_len=None, solution=None):

        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size, Name=Name, opt_len=opt_len)
            self.origin_problem = self.env.problems

            if self.env.test_in_tsplib:
                self.optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                            need_optimal=True)
                self.optimal_length, name = opt_len, Name
            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'TSP' + str(self.origin_problem.shape[1])
            B_V = batch_size * 1

            problem_size = self.origin_problem.shape[1]

            # notice that different baselines may have different formats
            best_select_node_list = torch.tensor(solution, device=self.origin_problem.device).unsqueeze(0)
            # best_select_node_list=torch.tensor(solution).unsqueeze(0)
            # best_select_node_list = torch.tensor(solution.astype(int)).unsqueeze(0)
            if self.tester_params['PRC']:
                origin_problem = self.env.problems[:]
                num_RC = self.tester_params['num_PRC']
                if problem_size <= 100000:
                    sample_max = problem_size // 4
                    lower = 2
                else:
                    sample_max = problem_size // 10
                    lower = problem_size // 10000

                for step_RC in range(num_RC):
                    val_num_samples = torch.randint(low=lower, high=sample_max + 1, size=[1])[0]
                    max_lenth = problem_size // val_num_samples
                    interval = problem_size // val_num_samples
                    if step_RC % 2 != 0:
                        best_select_node_list = torch.flip(best_select_node_list, dims=[1])
                    best_select_node_list = best_select_node_list.roll(dims=1, shifts=int(
                        torch.randint(low=0, high=problem_size, size=[1])[0]))
                    if_rotation = torch.randint(low=0, high=8, size=[1])[0]
                    if if_rotation != 0:
                        x = origin_problem[:, :, [0]]
                        y = origin_problem[:, :, [1]]
                        if if_rotation == 1:
                            origin_problem = torch.cat((1 - x, y), dim=2)
                        elif if_rotation == 2:
                            origin_problem = torch.cat((x, 1 - y), dim=2)
                        elif if_rotation == 3:
                            origin_problem = torch.cat((1 - x, 1 - y), dim=2)
                        elif if_rotation == 4:
                            origin_problem = torch.cat((y, x), dim=2)
                        elif if_rotation == 5:
                            origin_problem = torch.cat((1 - y, x), dim=2)
                        elif if_rotation == 6:
                            origin_problem = torch.cat((y, 1 - x), dim=2)
                        elif if_rotation == 7:
                            origin_problem = torch.cat((1 - y, 1 - x), dim=2)

                    select_node_list = best_select_node_list[:]
                    indices = torch.arange(0, problem_size, step=interval, dtype=torch.long)[:val_num_samples]
                    if problem_size <= 100000:
                        len_of_sub = torch.randint(low=4, high=max_lenth + 1, size=[1])[0]
                    else:
                        high_max = int(31622 / (val_num_samples ** 0.5))
                        len_of_sub = torch.randint(low=4, high=min(high_max, max_lenth) + 1, size=[1])[0]
                    new_problem, new_solution = self.sampling_subpaths_p(origin_problem, select_node_list, indices,
                                                                         len_of_sub)
                    self.env.problems = new_problem.view(-1, len_of_sub, 2)
                    self.env.solution = new_solution.view(-1, len_of_sub)
                    self.env.batch_size = self.env.problems.size(0)
                    partial_solution_length = self.env._get_travel_distance_2(self.env.problems, self.env.solution)
                    self.env.reset()
                    state, _, _, done = self.env.pre_step()
                    self.model.pre_forward(state=state)
                    current_step = 0
                    while not done:
                        if current_step == 0:
                            selected_teacher = self.env.solution[:, -1]
                            selected_student = self.env.solution[:, -1]
                        elif current_step == 1:
                            selected_teacher = self.env.solution[:, 0]
                            selected_student = self.env.solution[:, 0]
                        else:
                            selected_teacher, _, _, selected_student = self.model(state, self.env.selected_node_list,
                                                                                  self.env.solution, current_step,
                                                                                  repair=True)

                        current_step += 1
                        state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)
                    ahter_repair_sub_solution = torch.roll(self.env.selected_node_list, shifts=-1, dims=1)
                    after_repair_complete_solution = self.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                                                                            partial_solution_length,
                                                                                            reward_student, indices,
                                                                                            len_of_sub,
                                                                                            select_node_list.clone())

                    best_select_node_list = after_repair_complete_solution[:]
                    current_best_length = self.env._get_travel_distance_2(origin_problem, best_select_node_list)

            print('Get first complete solution!')
            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100

            self.logger.info("PRC, name:{}, gap:{:5f} %,  Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size

    def sampling_subpaths_p(self, problems, solution, indices, len_of_sub):
        batch_size, problems_size, embedding_size = problems.shape
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        new_sulution = solution[:, indices]
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        index_2, _ = new_sulution_ascending.type(torch.long).sort(dim=-1, descending=False)  # shape: [B, 2current_step]
        index_2 = index_2.view(batch_size, -1)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[
            1])
        new_data = problems[index_1, index_2].view(batch_size, -1, len_of_sub, 2)
        return new_data, new_sulution_rank

    def decide_whether_to_repair_solution(self, after_repair_sub_solution, before_reward, after_reward,
                                          indices, len_of_sub, solution):
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        origin_sub_solution = solution[:, indices]
        jjj, _ = torch.sort(origin_sub_solution, dim=-1, descending=False)
        kkk_2 = jjj.gather(2, after_repair_sub_solution.view(jjj.shape))
        if_repair = before_reward > after_reward
        if_repair = if_repair.unsqueeze(1).view(jjj.shape[0], jjj.shape[1])
        temp_result = solution[:, indices].clone()
        temp_result[if_repair] = kkk_2[if_repair]
        solution[:, indices] = temp_result
        return solution
