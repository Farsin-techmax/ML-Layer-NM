import os
import re

# Prepare retrain.py
content = open('retrain_20k.py').read()
content = content.replace('MILESTONE = 20', 'import sys\nMILESTONE = int(sys.argv[1])')

save_code = '''
        metrics_dict = {'Accuracy': float(acc), 'ROC-AUC': float(auc), 'F1-Score': float(f1), 'Precision': float(prec), 'Recall': float(rec), 'Brier': float(brier_s)}
        import json
        with open(os.path.join(MODEL_DIR, f'metrics_{label.replace(" ", "_")}.json'), 'w') as mf:
            json.dump(metrics_dict, mf)
'''
content = content.replace('        logger.info(f"Confusion Matrix:\\n{cm}")', '        logger.info(f"Confusion Matrix:\\n{cm}")\n' + save_code)

open('retrain.py', 'w').write(content)


# Prepare eval_plots.py
content2 = open('generate_eval_plots.py').read()
content2 = content2.replace('milestone = 20', 'import sys\n    milestone = int(sys.argv[1])')
open('generate_eval_plots.py', 'w').write(content2)

