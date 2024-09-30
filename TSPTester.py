from logging import getLogger
import torch

from TSP_Model import TSPModel as Model
from TSPEnv import TSPEnv as Env
from utils.utils import *
from utils.beam_search import Beamsearch


class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
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

    def run(self, size=None, disribution=None):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        self.env.load_raw_data(self.tester_params['test_episodes'], SL_Test=True, MSV=True, size=size,
                               disribution=disribution)

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student, problems_size = self._test_one_batch(episode, batch_size,
                                                                               clock=self.time_estimator_2)

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student, ))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

        return score_AM.avg, score_student_AM.avg, gap_

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

    def _test_one_batch(self, episode, batch_size, clock=None):
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            name = 'TSP' + str(self.origin_problem.shape[1])
            B_V = batch_size * 1
            problem_size = self.origin_problem.shape[1]

            # greedy
            current_step = 0
            reset_state, _, _ = self.env.reset()
            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node
            self.model.pre_forward(state=state)
            while not done:
                if current_step == 0:
                    selected_teacher = torch.zeros(B_V, dtype=torch.int64)
                    selected_student = selected_teacher
                else:
                    selected_teacher, _, _, selected_student = self.model(
                        state, self.env.selected_node_list, self.env.solution, current_step, )
                current_step += 1
                state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

            best_select_node_list = self.env.selected_node_list
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            torch.cuda.empty_cache()

            # beam
            if self.tester_params['beam']:
                beam_size = self.tester_params['beam_size']
                beamsearch = Beamsearch(beam_size, batch_size, problem_size, self.dtypeFloat, self.dtypeLong,
                                        probs_type='logits', random_start=False, device=self.origin_problem.device)
                current_step = 0
                self.env.reset(batch_size * beam_size)
                state, reward, reward_student, done = self.env.pre_step()
                while not done:
                    if current_step == 0:
                        selected_teacher = torch.zeros(batch_size * beam_size, dtype=torch.int64)
                    else:
                        _, trans_probs, _, _ = self.model(
                            state, self.env.selected_node_list, self.env.solution, current_step, beam_search=True,
                            beam_size=beam_size)
                        probs = torch.log(trans_probs.view(batch_size, beam_size, -1))
                        probs[probs.isnan()] = 0
                        self.env.selected_node_list = beamsearch.advance(probs, self.env.selected_node_list)
                        selected_teacher = beamsearch.next_nodes[-1].view(-1)
                    state, reward, done = self.env.step_beam(selected_teacher, beam=beam_size)
                    current_step += 1
                reward = reward.view(batch_size, beam_size)
                selected_node_list = self.env.selected_node_list.view(batch_size, beam_size, -1)
                current_best_length, min_idx = reward.min(1)
                zero_to_bsz = torch.arange(batch_size, dtype=torch.long, device=current_best_length.device)
                best_select_node_list = selected_node_list[zero_to_bsz, min_idx]
            torch.cuda.empty_cache()

            # PRC
            if self.tester_params['PRC']:
                origin_problem = self.env.problems[:]
                num_RC = self.tester_params['num_PRC']
                sample_max = problem_size // 4
                for step_RC in range(num_RC):
                    val_num_samples = torch.randint(low=2, high=sample_max + 1, size=[1])[0]
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
                    len_of_sub = torch.randint(low=4, high=max_lenth + 1, size=[1])[0]
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
            self.logger.info("greedy, name:{}, gap:{:4f} %,  Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            ####################################################

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size

    def sampling_subpaths_p(self, problems, solution, indices, len_of_sub):
        batch_size, problems_size, embedding_size = problems.shape
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        new_sulution = solution[:, indices]
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        index_2, _ = new_sulution_ascending.type(torch.long).sort(dim=-1, descending=False)
        index_2 = index_2.view(batch_size, -1)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[
            1])
        new_data = problems[index_1, index_2].view(batch_size, -1, len_of_sub, 2)
        return new_data, new_sulution_rank
