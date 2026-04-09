from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from matplotlib import pyplot
import os

def AUROC(scores, labels, output_dir):
    
    pyplot.ioff() 
    
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    ns_probs = [0 for _ in range(len(labels))]
    lr_auc = roc_auc_score(labels, scores)
    write_to_out('AUROC: %.3f \n' % (lr_auc), output_dir)
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(labels, scores)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, label='Logistic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    save_path = os.path.join(output_dir, 'AUROC.png')
    pyplot.savefig(save_path, dpi=180)
    pyplot.show()
    pyplot.clf()

def AUPR(scores, labels, output_dir):
    pyplot.ioff() 
    
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    lr_precision, lr_recall, _ = precision_recall_curve(labels, scores)
    lr_auc = auc(lr_recall, lr_precision)
    write_to_out('AUPR: %.3f \n' % (lr_auc), output_dir)
    no_skill = len(labels[labels==1]) / len(labels)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, label='HGT')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    save_path = os.path.join(output_dir, 'AUPR.png')
    pyplot.savefig(save_path, dpi=180)
    pyplot.show()
    pyplot.clf()
    
def plot_losses(losses, val_losses, output_dir):
    pyplot.plot(range(len(losses)), losses, label="loss")
    pyplot.plot(range(len(losses)), val_losses, label="val_loss")
    pyplot.legend()
    pyplot.savefig(output_dir+'/losses', dpi=200)
    pyplot.clf()
    
def write_to_out(text, output_dir):
    print(text)
    save_path = os.path.join(output_dir, 'out.txt')
    with open(save_path, 'a', encoding='utf-8') as out_file:
        out_file.write(text)