# from pathlib import Path
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc

# from torch.utils.data import DataLoader
# import torch

# from model import DiagnosticModel



# # def generate_roc(probs: list, labels: list, fpath, title: str | None = None, x_range: tuple[int, int] = (0,1)):
# #     """
# #     Outputs are the generated outputs thus far:
# #     0 = good
# #     1 = bad

# #     This generates the ROC curve and relevant statistics
# #     """
    
# #     xs = []
# #     ys = []

# #     for t in np.append(np.linspace(0, 1, 100), 0.5):
# #         preds = (probs >= t).int()
    
# #         tn, fp, fn, tp = conf_matrix(preds, labels)
    
# #         TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # given positive label, Prob correct?
# #         FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # given negative label, Prob wrong?

# #         xs.append(FPR)
# #         ys.append(TPR)

# #     # Create Plot
# #     xs, ys = zip(*sorted(zip(xs, ys)))
# #     roc_auc = auc(xs, ys)

# #     if fpath is None:
# #         return roc_auc

# #     plt.figure(figsize=(8, 6))
# #     plt.plot(xs, ys, 'b-', linewidth=2, label=f'ROC Curve')
# #     plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')
# #     if title is not None:
# #         plt.title(f'{title}: ROC Curve (AUC = {roc_auc:.3f})')
# #     else:
# #         plt.title(f'{title}: ROC Curve (AUC = {roc_auc:.3f})')
# #     plt.legend(loc='lower right')
# #     plt.grid(True, alpha=0.3)
# #     plt.xlim(x_range)
# #     plt.ylim([0.0, 1.0])
# #     plt.tight_layout()
# #     plt.savefig(fpath)
# #     plt.close()

# #     return roc_auc
    

# # def conf_matrix(preds, labels) -> np.array:
# #     """
# #     Compute confusion matrix for binary classification. Preds
# #     and labels can be any Tensor as long as they are the same size!
    
# #     Returns:
# #     tn, fp, fn, tp: Confusion matrix counts.
# #     """
# #     assert preds.shape == labels.shape
# #     # Ensure they are ints!
# #     labels = labels.int()
# #     preds = preds.int()

# #     # Compute values
# #     tp = ((labels == 1) & (preds == 1)).sum().item()
# #     tn = ((labels == 0) & (preds == 0)).sum().item()
# #     fp = ((labels == 0) & (preds == 1)).sum().item()
# #     fn = ((labels == 1) & (preds == 0)).sum().item()
    
# #     return np.array([tn, fp, fn, tp])

# # def print_accuracies(epoch: int, num_epochs: int, loss: float, train_cfvalues, val_cfvalues, fname: Path | None = None):
# #      text = (
# #          f"Epoch {epoch+1}/{num_epochs}\n" +
# #          f"  Loss = {loss:.4f}\n" + 
# #          f"  Train: {get_info_str(train_cfvalues)}\n" + 
# #          f"  Val:   {get_info_str(val_cfvalues)}\n"
# #         )
     
# #      if fname is not None:
# #         write_type = 'w' if epoch == 0 else 'a' # append or rewrite

# #         with open(fname, write_type) as f:  
# #             f.write(text)

# # def get_info(cf_values: list) -> dict:
# #     cfv = cf_values / sum(cf_values)

# #     info = {}
# #     info['tn'] = cfv[0]
# #     info['fp'] = cfv[1]
# #     info['fn'] = cfv[2]
# #     info['tp'] = cfv[3]

# #     info['tpr'] = info['tp'] / (info['tp'] + info['fn'])
# #     info['fpr'] = info['fp'] / (info['fp'] + info['tn'])

# #     info['prec'] = info['tp'] / (info['tp'] + info['fp'])
# #     info['recall'] = info['tp'] / (info['tp'] + info['fn'])
# #     info['f1'] = 2 * (info['recall'] * info['prec']) / (info['recall'] + info['prec'])
# #     info['acc'] = info['tn'] + info['tp']

# #     return info

# # def get_info_str(cf_values: list) -> str:
# #     """
# #     results: list of (outputs, labels)
# #     """

# #     info = get_info(cf_values)
# #     return f"""
# #         tn = {info['tn']*100:4.2f}, tp = {info['tp']*100:4.2f}
# #         fn = {info['fn']*100:4.2f}, fp = {info['fp']*100:4.2f}

# #         prec = {info['prec']*100:4.2f}, recall = {info['recall']*100:4.2f}
# #         tpr = {info['tpr']*100:4.2f}, fpr = {info['fpr']*100:4.2f}
# #         f1 = {info['f1']*100:4.2f}, acc = {info['acc']*100:4.2f}
# #     """

# # def display_curve(train_full: list[np.ndarray], val_full: list[np.ndarray], 
# #                   loss_full: list[float], loss_val_full: list[float],
# #                   full_val_auc: list[float],
# #                   dir: Path, 
# #                   title: str,
# #                   metrics: list[str], 
# #                   colors: list[str], 
# #                   val_metric: str):
# #     """
# #     Plots Accuracy, FPR, FNR over epochs.
# #     Each entry in train_cfvalues / val_cfvalues is [TN, FP, FN, TP].
# #     """
# #     train_info = [get_info(cfv) for cfv in train_full]
# #     val_info = [get_info(cfv) for cfv in val_full]

# #     # Plot
# #     epochs = range(1, len(train_full) + 1)  
# #     num_ticks_displayed = 10    
# #     tick_jump = math.ceil(len(epochs) / num_ticks_displayed)   
# #     fig, ax1 = plt.subplots(figsize=(8, 5))

# #     # ---- METRICS ----
# #     for metric, color in zip(metrics, colors):
# #         if metric == 'loss' or metric == 'val_loss':
# #             continue
# #         if metric == 'auc':
# #             ax1.plot(epochs, full_val_auc, 
# #                      label="Val AUC",
# #                      color = color, linestyle='-')
# #             continue

# #         ax1.plot(epochs, [dic[metric] for dic in train_info],
# #                  label=f"Train {metric.upper()}",
# #                  color=color, linestyle="--")
# #         ax1.plot(epochs, [dic[metric] for dic in val_info],
# #                  label=f"Val {metric.upper()}",
# #                  color=color, linestyle="-")

# #     ax1.set_xlabel("Epoch")
# #     ax1.set_ylabel("Metric")
# #     ax1.set_ylim(0.0, 1.0)
# #     ax1.set_xticks(ticks = epochs[::tick_jump], labels = epochs[::tick_jump])
# #     ax1.tick_params(axis='x', labelsize=8)
# #     ax1.grid(True)
# #     ax1.legend()

# #     # Highlighting Max Validation Metric on Ax1
# #     vms = [dic[val_metric] for dic in val_info]
# #     max_vm_idx = np.argmax(vms)
# #     max_vm = np.max(vms)
# #     ax1.scatter(epochs[max_vm_idx], max_vm, color='black', s=20, zorder=5)  # black dot
# #     ax1.text(epochs[max_vm_idx], max_vm + 0.02, f"{max_vm:.3f}", color='black', ha='center')

# #     fig.tight_layout()
# #     fig.savefig(dir / f'lc_metrics_{title[:3]}.png', dpi=150)
# #     plt.close(fig)

# #     # ---- LOSS! ----
# #     fig2, ax2 = plt.subplots(figsize = (8,5))
# #     ax2.plot(epochs, loss_full, label="Loss", color="blue", linewidth=2, linestyle='--')
# #     ax2.plot(epochs, loss_val_full, label="Val Loss", color="blue", linewidth=2, linestyle='-')
# #     ax2.legend()
# #     ax2.set_xlabel("Epoch")
# #     ax2.set_ylabel("Loss")
# #     ax2.set_xticks(ticks = epochs[::tick_jump], labels = epochs[::tick_jump])
# #     ax2.tick_params(axis='x', labelsize=8)
# #     ax2.grid(True)
# #     ax2.set_title(f"Losses for {title}")

# #     fig2.tight_layout()
# #     fig2.savefig(dir / f'lc_loss_{title[:3]}.png', dpi=500)
# #     plt.close(fig2)


# def save_bad_examples(model: DiagnosticModel, data_loader: DataLoader, output_dir: Path, ckpt_path: Path = None):
#     """
#     Takes in a model and a data_loader, and it saves a file that shows mis-classifications
#     """
#     device = next(model.parameters()).device
#     if ckpt_path is not None:
#         checkpoint = torch.load(ckpt_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])

#     model.eval()

#     all_fp = []
#     all_fn = []

#     fp_indices = []
#     fn_indices = []

#     global_idx = 0   # <-- Track index in dataset

#     for data, labels in data_loader:
#         # Dealing w/ idxs
#         batch_size = labels.size(0)
#         batch_indices = np.arange(global_idx, global_idx + batch_size)
#         global_idx += batch_size

#         # Running the Model
#         data, labels = data.to(device), labels.to(device)

#         outputs = model(data) 
#         _, preds = torch.max(outputs, dim = 1)

#         false_pos_mask = (labels == 0) & (preds == 1)
#         false_neg_mask = (labels == 1) & (preds == 0)

#         # Save images (move channels to the end; extend the list by batch)
#         all_fp.extend(data[false_pos_mask].cpu().permute(0, 2, 3, 1))  # each img is (W, H, 3)
#         all_fn.extend(data[false_neg_mask].cpu().permute(0, 2, 3, 1))

#         # Save idxs
#         fp_indices.extend(batch_indices[false_pos_mask.cpu().numpy()])
#         fn_indices.extend(batch_indices[false_neg_mask.cpu().numpy()])


#     model.train()

#     # ---- SHUFFLE PAIRS TOGETHER ----

#     fp_perm = np.random.permutation(len(all_fp))
#     fn_perm = np.random.permutation(len(all_fn))

#     all_fp = [all_fp[i] for i in fp_perm]
#     fp_indices = [fp_indices[i] for i in fp_perm]

#     all_fn = [all_fn[i] for i in fn_perm]
#     fn_indices = [fn_indices[i] for i in fn_perm]

#     # ---- DISPLAY FALSE POSITIVES ----

#     print(f"There are {len(all_fp)} false positives")
#     fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
#     for i, ax in enumerate(axes.flat):
#         if i >= len(all_fp):
#             break
#         ax.imshow(all_fp[i][:, :, 0], cmap='grey', vmin=0, vmax=1)
#         ax.set_title(f"idx={fp_indices[i]}")
#         ax.axis('off')
#     fig.suptitle("False Positives (good, but predicted bad)", fontsize=20)
#     plt.tight_layout()
#     fig.savefig(output_dir / "false_positives.png")
#     plt.close(fig)

#     # ---- DISPLAY FALSE NEGATIVES ----
    
#     print(f"There are {len(all_fn)} false negatives")
#     fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
#     for i, ax in enumerate(axes.flat):
#         if i >= len(all_fn):
#             break
#         ax.imshow(all_fn[i][:, :, 0], cmap='grey', vmin=0, vmax=1)
#         ax.set_title(f"idx={fn_indices[i]}")
#         ax.axis('off')
#     fig.suptitle("False Negatives (bad, but predicted good)", fontsize=20)
#     plt.tight_layout()
#     fig.savefig(output_dir / "false_negatives.png")
#     plt.close(fig)

#     return fp_indices, fn_indices

