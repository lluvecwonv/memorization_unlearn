from omegaconf import OmegaConf
import hydra 
import json
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
import os

def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Prob.', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[metric + ' ' + eval_task_dict[eval_task]] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'Prob. {eval_task_dict[k]}'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'ROUGE {eval_task_dict[k]}'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))

        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        output_result[f'Truth Ratio {eval_task_dict[k]}'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

@hydra.main(version_base=None, config_path="../config", config_name="aggregate_eval_stat")
def main(cfg):
    if cfg.retain_result is None or cfg.ckpt_result is None:
        raise ValueError("Please provide both retain_result and ckpt_result")
    
    print(f"Loading retain results from: {cfg.retain_result}")
    print(f"Loading checkpoint results from: {cfg.ckpt_result}")
    
    retain_result = json.load(open(cfg.retain_result))
    ckpt_result = json.load(open(cfg.ckpt_result))

    # We have to assume here that retain_result and ckpt_result follow these structure:
    # The top most layer has ['eval_log.json', 'eval_log_forget.json', 'eval_real_world_wo_options.json', 'eval_real_author_wo_options']
    # the second layer contains the actual metrics: ['avg_gt_loss', 'average_perturb_loss', 'avg_paraphrased_loss', 'rougeL_recall']
    # within each metric, we have {data_idx: measurement}

    print("Calculating model utility...")
    model_utility = get_model_utility(ckpt_result)
    
    print("Calculating forget quality...")
    forget_quality = get_forget_quality(ckpt_result, retain_result)
    model_utility.update(forget_quality)  # Add all forget quality metrics

    # Add metadata
    model_utility['Method'] = cfg.method_name
    model_utility['Submitted By'] = getattr(cfg, 'submitted_by', 'Unknown')
    
    # Print results summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Method: {model_utility['Method']}")
    print(f"Model Utility: {model_utility['Model Utility']:.4f}")
    print(f"Forget Quality: {model_utility['Forget Quality']:.4f}")
    print(f"KS Test Statistic: {model_utility['KS Test Forget']:.4f}")
    print("="*50)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(cfg.save_file), exist_ok=True)
    
    # Determine file extensions
    if cfg.save_file.endswith('.json'):
        json_file = cfg.save_file
        csv_file = cfg.save_file.replace('.json', '.csv')
    elif cfg.save_file.endswith('.csv'):
        csv_file = cfg.save_file
        json_file = cfg.save_file.replace('.csv', '.json')
    else:
        # Default to JSON if no extension
        json_file = cfg.save_file + '.json'
        csv_file = cfg.save_file + '.csv'
    
    # Save as JSON for better readability
    with open(json_file, 'w') as f:
        json.dump(model_utility, f, indent=2, default=str)
    print(f"Results saved to JSON: {json_file}")
    
    # Also save as CSV for compatibility
    with open(csv_file, 'w', newline='') as f:
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
    print(f"Results saved to CSV: {csv_file}")
    
    # Pretty print all metrics
    print("\nDetailed Results:")
    pprint.pprint(model_utility)
    
    return model_utility

if __name__ == "__main__":
    main()