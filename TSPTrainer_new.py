from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from TSP_Model import TSPModel as Model
from TSPEnv import TSPEnv as Env
from utils.utils import *
from utils.beam_search import Beamsearch


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.dtypeFloat = torch.cuda.FloatTensor
        self.dtypeLong = torch.cuda.LongTensor

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']  # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        random_seed = 2024
        torch.manual_seed(random_seed)
        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)

        model_load = trainer_params['model_load_path']
        model_epoch = trainer_params['model_load_epoch']
        checkpoint_fullname = '{}/checkpoint-{}.pt'.format(model_load, model_epoch)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved Model Loaded !!')
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)
        self.env.load_raw_data(1000000, SL_Test=False)
        problem_size_init = self.trainer_params['problem_size_init']
        problem_size_max = self.trainer_params['problem_size_max']
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')
            # curriculum learning
            problem_size = problem_size_init + epoch * (problem_size_max - problem_size_init) // self.trainer_params[
                'epochs']

            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch, problem_size=problem_size)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, problem_size=100):

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        self.env.load_problems_4_each_epoch(self.trainer_params['train_episodes'], problem_size)

        # Greedy
        greedy_values, greedy_list = self.validation_greedy()

        # Beam
        beam_values, beam_list = self.validation_beam(problem_size=problem_size)
        index_update = greedy_values < beam_values
        beam_list[index_update] = greedy_list[index_update]

        # PRC
        PRC_value, all_list = self.PRC(beam_list)

        torch.cuda.empty_cache()
        best_values = PRC_value
        self.env.raw_data_tours = all_list
        iterations = 0
        per_batch = self.trainer_params['per_batch']
        max_limit = self.trainer_params['max_limit']
        best_limit = self.trainer_params['best_limit']
        while iterations < max_limit and (greedy_values.mean() - best_values) / best_values > 0.001 and best_limit > 0:
            if best_values > PRC_value:
                best_values = PRC_value
                self.env.raw_data_tours = all_list
                best_limit = self.trainer_params['best_limit']
                best_limit -= 1
            else:
                best_limit -= 1
            self.logger.info('greedy_values: {:3f}, best_values: {:.3f}'.format(greedy_values.mean(), best_values))
            self.logger.info(
                'iteration: {:2d}, gap: {:.3f}'.format(iterations, (greedy_values.mean() - best_values) / best_values))
            for _ in range(per_batch):
                while episode < train_num_episode:
                    remaining = train_num_episode - episode
                    batch_size = min(self.trainer_params['train_batch_size'], remaining)
                    avg_score, score_student_mean, avg_loss = self._train_one_batch(episode, batch_size, mix=True)
                    torch.cuda.empty_cache()
                    score_AM.update(avg_score, batch_size)
                    score_student_AM.update(score_student_mean, batch_size)
                    loss_AM.update(avg_loss, batch_size)

                    episode += batch_size

                    loop_cnt += 1
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                            .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                    score_AM.avg, score_student_AM.avg, loss_AM.avg))
                self.env.shuffle_data()
                episode = 0

            greedy_values, greedy_list = self.validation_greedy()
            beam_values, beam_list = self.validation_beam(problem_size=problem_size)
            index_update = greedy_values < beam_values
            beam_list[index_update] = greedy_list[index_update]
            PRC_value, all_list = self.PRC(beam_list)
            torch.cuda.empty_cache()
            iterations += 1

            # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg))
        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    @torch.no_grad()
    def PRC(self, all_list, num_RC=1000):
        self.model.eval()
        val_num_episode = self.trainer_params['train_episodes']
        problem_size = self.env.raw_data_nodes.size(1)
        sample_max = problem_size // 4
        for step_RC in range(num_RC):
            val_num_samples = torch.randint(low=2, high=sample_max + 1, size=[1])[0]
            max_lenth = problem_size // val_num_samples
            interval = problem_size // val_num_samples
            if step_RC % 2 != 0:
                all_list = torch.flip(all_list, dims=[1])
            all_list = all_list.roll(dims=1, shifts=int(torch.randint(low=0, high=problem_size, size=[1])[0]))
            all_node_list = None
            all_value = None
            episode = 0
            while episode < val_num_episode:
                remaining = val_num_episode - episode
                batch_size = min(self.trainer_params['val_beam_batch_size'], remaining)
                if_rotation = torch.randint(low=0, high=8, size=[1])[0]
                self.env.load_problems_val(episode, batch_size, if_rotation)
                origin_problem = self.env.problems
                select_node_list = all_list[episode:episode + batch_size]
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
                        selected_teacher = self.env.solution[:, -1]  # destination node
                        selected_student = self.env.solution[:, -1]
                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]  # starting node
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

                all_node_list = torch.cat((all_node_list, after_repair_complete_solution),
                                          dim=0) if all_node_list is not None else after_repair_complete_solution
                current_best_length = self.env._get_travel_distance_2(origin_problem, after_repair_complete_solution)
                all_value = torch.cat((all_value, current_best_length),
                                      dim=0) if all_value is not None else current_best_length
                episode += batch_size
            all_list = all_node_list[:]
        return all_value.mean(), all_list

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

    def sampling_subpaths_p(self, problems, solution, indices, len_of_sub):
        batch_size, problems_size, embedding_size = problems.shape
        indices = indices.unsqueeze(1) + torch.arange(len_of_sub)
        new_sulution = solution[:, indices]
        new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)
        _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
        index_2, _ = new_sulution_ascending.type(torch.long).sort(dim=-1, descending=False)
        index_2 = index_2.view(batch_size, -1)
        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])
        new_data = problems[index_1, index_2].view(batch_size, -1, len_of_sub, 2)
        return new_data, new_sulution_rank

    @torch.no_grad()
    def validation_beam(self, problem_size):
        self.model.eval()
        val_num_episode = self.trainer_params['train_episodes']
        all_value = None
        all_node_list = None
        episode = 0
        beam_size = self.trainer_params['beam_size']
        while episode < val_num_episode:
            remaining = val_num_episode - episode
            batch_size = min(self.trainer_params['val_beam_batch_size'], remaining)
            initial_value, initial_list = self.beamsearch_tour_nodes_shortest(beam_size, batch_size, problem_size,
                                                                              self.dtypeFloat, self.dtypeLong,
                                                                              episode, probs_type='logits',
                                                                              random_start=False)
            all_value = torch.cat((all_value, initial_value), dim=0) if all_value is not None else initial_value
            all_node_list = torch.cat((all_node_list, initial_list),
                                      dim=0) if all_node_list is not None else initial_list
            episode += batch_size
        return all_value, all_node_list

    @torch.no_grad()
    def beamsearch_tour_nodes_shortest(self, beam_size, batch_size, num_nodes,
                                       dtypeFloat, dtypeLong, episode, probs_type='raw', random_start=False,
                                       device=torch.device('cuda')):
        beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start,
                                device=device)
        self.env.load_problems_val(episode, batch_size)
        self.env.reset(batch_size * beam_size)
        state, _, _, done = self.env.pre_step()  # state: data, first_node = current_node
        self.model.pre_forward(state=state)
        current_step = 0
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
        min_values, min_idx = reward.min(1)
        zero_to_bsz = torch.arange(batch_size, dtype=torch.long, device=min_values.device)
        min_selected_node_list = selected_node_list[zero_to_bsz, min_idx]
        return min_values, min_selected_node_list

    @torch.no_grad()
    def validation_greedy(self):
        self.model.eval()
        val_num_episode = self.trainer_params['train_episodes']
        all_value = None
        all_node_list = None
        episode = 0
        with torch.no_grad():
            while episode < val_num_episode:
                remaining = val_num_episode - episode
                batch_size = min(self.trainer_params['val_batch_size'], remaining)
                initial_value, initial_list = self._val_batch_greedy(episode, batch_size)
                all_value = torch.cat((all_value, initial_value), dim=0) if all_value is not None else initial_value
                all_node_list = torch.cat((all_node_list, initial_list),
                                          dim=0) if all_node_list is not None else initial_list
                episode += batch_size
        return all_value, all_node_list

    @torch.no_grad()
    def _val_batch_greedy(self, episode, batch_size):
        self.env.load_problems_val(episode, batch_size)
        self.env.reset(batch_size)
        state, _, _, done = self.env.pre_step()  # state: data, first_node = current_node
        self.model.pre_forward(state=state)
        current_step = 0
        B_V = batch_size * 1
        while not done:
            if current_step == 0:
                selected_teacher = torch.zeros(B_V, dtype=torch.int64)
                selected_student = selected_teacher
                state, _, _, done = self.env.step(selected_teacher, selected_student)
            else:
                selected_teacher, _, _, selected_student = self.model(
                    state, self.env.selected_node_list, self.env.solution, current_step, )
                state, _, _, done = self.env.step(selected_teacher, selected_student)

            current_step += 1
        best_select_node_list = self.env.selected_node_list
        current_best_length = self.env._get_travel_distance_2(self.env.problems, best_select_node_list)
        return current_best_length, best_select_node_list

    def _train_one_batch(self, episode, batch_size, mix=False):

        ###############################################
        self.model.train()
        self.env.load_problems(episode, batch_size, mix, train=True)
        problem_size = self.env.problems.size(1)
        if mix:
            batch_size = self.env.batch_size

        self.env.reset()
        prob_list = torch.ones(size=(batch_size, 0))

        state, _, _, done = self.env.pre_step()

        self.model.pre_forward(state=state)

        current_step = 0

        while not done:
            if current_step == 0:
                selected_teacher = self.env.solution[:, -1]  # destination node
                selected_student = self.env.solution[:, -1]
                prob_copy = torch.ones(batch_size, 1)
            elif current_step == 1:
                selected_teacher = self.env.solution[:, 0]  # starting node
                selected_student = self.env.solution[:, 0]
                prob_copy = torch.ones(batch_size, 1)
            else:
                selected_teacher, prob, _, selected_student = self.model(state, self.env.selected_node_list,
                                                                         self.env.solution, current_step)
                prob_copy = prob
                if mix and problem_size - current_step > 98:
                    prob = prob[batch_size // 2:]
                    prob_copy = prob.repeat(2, 1)
                prob = prob[prob > 1e-3]
                loss_mean = -prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step += 1

            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)
            prob_list = torch.cat((prob_list, prob_copy), dim=1)

        loss_mean = -prob_list.log().mean()

        return 0, 0, loss_mean.item()
