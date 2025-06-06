import math
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

class ARGS:
    def __init__(self, num_users, num_items, latent_dim, learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        
class Evaluate:
    def __init__(self, testing_data, neg_testing, isShow=False, device='cuda', output_csv_path=None):
        self.user_num = len(testing_data)
        self.testing_data = testing_data
        self.neg_testing = neg_testing
        self.isShow = isShow
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_csv_path = output_csv_path
        self.results = {}

    def set_eva_function(self, predice_function):
        self.predice_function = predice_function

    def run(self, batch_size=256, model='bpr', epochs=1):
        if not hasattr(self, 'predice_function'):
            raise RuntimeError("Prediction function not set. Call set_eva_function() first.")

        topks = [1, 3, 5, 10]
        self.results = {metric: {k: 0.0 for k in topks} for metric in ['num_hit', 'MAP_score', 'NDCG_score']}
        user_records = []

        user_ids = list(self.testing_data.keys())
        all_batches = [user_ids[i:i+batch_size] for i in range(0, len(user_ids), batch_size)]

        for batch_uids in tqdm(all_batches, desc="Evaluating in batch"):
            batch_user_tensors = []
            batch_input_tensors = []
            batch_test_tensors = []

            for uid in batch_uids:
                input_seq = self.testing_data[uid][0]
                negatives = self.neg_testing[uid][:-1]
                test_items = negatives + [self.testing_data[uid][1]]

                # batch_user_tensors.append(torch.tensor([uid] * 100))
                # batch_input_tensors.append(torch.tensor(input_seq).unsqueeze(0))  # (1, seq_len)
                # batch_test_tensors.append(torch.tensor(test_items))
                for test_item in test_items:
                    batch_user_tensors.append(uid)
                    batch_input_tensors.append(input_seq)
                    batch_test_tensors.append(test_item)
                

            # # Stack all into a single batch
            # user_tensor = torch.cat(batch_user_tensors, dim=0).to(self.device)
            # input_tensor = torch.cat(batch_input_tensors, dim=0).to(self.device)
            # test_tensor = torch.cat(batch_test_tensors, dim=0).to(self.device)
            
            # Stack all into a single batch
            user_tensor = torch.tensor(batch_user_tensors, dtype=torch.long).to(self.device)        # [batch_size*100]
            input_tensor = torch.tensor(batch_input_tensors, dtype=torch.long).to(self.device)      # [batch_size*100, seq_len]
            test_tensor = torch.tensor(batch_test_tensors, dtype=torch.long).to(self.device)        # [batch_size*100]

            # Run prediction
            # print("user_tensor", user_tensor.shape)
            # print("input_tensor", input_tensor.shape)
            # print("test_tensor", test_tensor.shape)
            
            if model == 'bpr':
                scores, _ = self.predice_function(user_tensor, input_tensor, test_tensor, test_tensor)
            elif model == 'ncf':
                scores = self.predice_function(user_tensor, test_tensor)
            scores_np = scores.detach().cpu().numpy()

            # Slice scores per user
            for i, uid in enumerate(batch_uids):
                user_scores = scores_np[i * 100:(i + 1) * 100]
                true_score = user_scores[-1]
                rank = np.sum(user_scores >= true_score)
                true_rank = int(rank)

                record = {
                    "user_id": uid,
                    "true_rank": true_rank,
                }

                if self.isShow:
                    print(f"User {uid} | Rank: {true_rank}")

                for k in topks:
                    hit = 1 if true_rank <= k else 0
                    self.results['num_hit'][k] += hit
                    self.results['MAP_score'][k] += (1.0 / true_rank) if hit else 0.0
                    self.results['NDCG_score'][k] += (math.log(2.0) / math.log(true_rank + 1)) if hit else 0.0
                    record[f"hit@{k}"] = hit

                user_records.append(record)

        for metric in self.results:
            for k in topks:
                self.results[metric][k] /= self.user_num

        if self.output_csv_path:
            output_csv_path = self.output_csv_path.replace('.csv', f'_{model}_eval_ep_{epochs}.csv')
            df = pd.DataFrame(user_records)
            # df.to_csv(output_csv_path, index=False)
            # print(f"📄 Saved batch eval result to {output_csv_path}")
        
            return self.results, df
        return self.results, None


    def print_result(self):
        result = "\nEvaluate:\n"
        metrics = ['num_hit', 'MAP_score', 'NDCG_score']
        topks = [1, 3, 5, 10]

        for metric in metrics:
            result += f"{metric}:\t" + "\t".join([f"@{k}" for k in topks]) + "\n\t\t"
            print(f"{metric}:\t" + "\t".join([f"@{k}" for k in topks]), end="\n\t\t")
            for k in topks:
                val = self.results[metric][k]
                result += f"{val:.4f}\t"
                print(f"{val:.4f}", end="\t")
            result += "\n"
            print()
        return result

def make_training_data(user_session, window_size=5):
    print("Generating training data...")
    one_session_len = window_size + 1
    train_data = []
    test_data = []
    test_dict = {}

    for u, items in tqdm(user_session.items()):
        session_len = len(items)
        user_data = []

        for i in range(session_len - one_session_len + 1):
            input_seq = items[i:i + window_size]
            target = items[i + window_size]
            user_data.append((u, input_seq, target))

        if len(user_data) < 1:
            continue  # 防止太短的session

        train_data.extend(user_data[:-1])
        test_data.append(user_data[-1])
        test_dict[user_data[-1][0]] = (user_data[-1][1], user_data[-1][2])

    return train_data, test_dict
